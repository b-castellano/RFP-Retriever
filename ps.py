import openai
import pandas as pd
import re
import threading
import json
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import FAQPipeline
from langchain.prompts import PromptTemplate
import utils

def init():
    # Init docstore, retriever, write docs, gpt, return a pipeline for doc search
    document_store, loaded = init_store()
    retriever = init_retriever(document_store)
    if not loaded:
        write_docs(document_store, retriever)
    init_gpt()
    return init_pipe(retriever)

# Initiate FAISS datastore
def init_store():
    try:
        return FAISSDocumentStore.load(index_path="my_faiss_index.faiss"), True
    except:
        return FAISSDocumentStore(
            similarity="cosine",
            embedding_dim=768,
            duplicate_documents='overwrite'
        ), False
    
# Initiate retriever for documents
def init_retriever(document_store):
    return EmbeddingRetriever(
        document_store=document_store,
        embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
        model_format="sentence_transformers"
    )

# Write documents in datastore if not already
def write_docs(document_store, retriever):
    # Get dataframe with columns "question", "answer" and some custom metadata
    df = pd.read_csv("qna.csv")
    df.fillna(value="", inplace=True)

    # Create embeddings for our questions from the FAQs
    questions = list(df["question"].values)
    print("questions:", len(questions))
    df["embedding"] = retriever.embed_queries(queries=questions).tolist()
    df = df.rename(columns={"question": "content"})

    # Convert Dataframe to list of dicts and index them in our DocumentStore
    docs_to_index = df.to_dict(orient="records")
    print("dictionaries:", len(docs_to_index))
    document_store.write_documents(docs_to_index)
    document_store.update_embeddings(retriever)

    document_store.save(index_path="my_faiss_index.faiss")
    print("docs added:", document_store.get_document_count())
    print("docs embedded:", document_store.get_embedding_count())

# Initiate the pipeline
def init_pipe(retriever):
    return FAQPipeline(retriever=retriever)

# Get response for query
def get_response(pipe, query):
    prediction, closeMatch = query_faiss(query, pipe) 

    # Generate prompt from related docs
    prompt, scores, alts, CIDs, source_links, source_filenames, SMEs, best_sme = create_prompt(query, prediction)
    
    if closeMatch:
        newAnswer = re.sub("[\[\]'\"]","",prediction.meta["answer"])
        score = prediction.score * 100
        #source = prediction.meta["cid"]
        output = newAnswer
        conf = score
    else:
        output, conf = call_gpt(prompt, scores, alts)
    
    #output = re.sub("\.\s+\.", ".", output)

    return output, conf, CIDs, source_links, source_filenames, SMEs, best_sme

# Get top k documents related to query from datastore
def query_faiss(query, pipe):
    docs = pipe.run(query=query, params={"Retriever": {"top_k": 5}})

    # If there is a close match between user question and pulled doc, then just return that doc's answer
    print(docs["documents"])
    if docs["documents"][0].score > .95:
        return docs["documents"][0], True
        
    return docs, False

# Create prompt template
def create_prompt(query, prediction):  ## May add a parameter "Short", "Yes/No", "Elaborate", etc. for answer preferences
    print("Creating prompt")
    prompt = PromptTemplate(input_variables=["prefix", "question", "context"],
                            template="{prefix}\nQuestion: {question}\n Context: {context}\n")

    # Provide instructions/prefix
    prefix = """You are an assistant for the Information Security department of an enterprise designed to answer security questions professionally. Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, answer'. Use the answers within the context to answer the original question in a concise manner with explanation. Just at the end, list the question IDs of the answers you referenced to formulate your response."""
    
    # Create context
    context = ""
    scores = {}
    SMEs, sme_dict, source_filenames, source_links, CIDs = [], {}, [], [], []  # SMEs are dictionary in this function to allow us to more easily get best_sme later
    alts = []
    count = 0

    for answer in prediction["answers"]:
        newAnswer = re.sub("[\[\]'\"]","",answer.meta["answer"])

        # Remove docs 
        context += "Question ID: {ID}, Answer: {answer}, Date: {date}\n".format(
            ID=answer.meta["cid"], answer=newAnswer, date=answer.meta["updated date"])
        
        # Add ID-Score pair to dict
        scores[answer.meta["cid"]] = answer.score

        # If exists, update each array with corresponding metadata
        if answer.meta["cid"] not in [None, ""]:
            CIDs.append(answer.meta["cid"])
        else:
            CIDs.append("N/A")
        if answer.meta["url"] not in [None, ""]:
            source_links.append(answer.meta["url"])
        else:
            source_links.append("N/A")
        if answer.meta["sme"] not in [None, ""]:
            SMEs.append(answer.meta["sme"])
            sme_dict[answer.meta["cid"]] = answer.meta["sme"]
        else:
            SMEs.append("N/A")
            sme_dict[answer.meta["cid"]] = "N/A"
        if answer.meta["file name"] not in [None, ""]:
            source_filenames.append(answer.meta["file name"])
        else:
            source_filenames.append("N/A")
        if count < 3:
            alts.append(answer.meta["cid"])
            count+=1

    # Get the question ID with the highest score, use as primary SME
    max_id = max(scores, key=scores.get)
    best_sme = sme_dict[max_id]

    return prompt.format(prefix=prefix, question=query, context=context), scores, alts, CIDs, source_links, source_filenames, SMEs, best_sme 
    

# Call openai API
def call_gpt(prompt,scores, alts):  # returns None as confidence if no sources used for prompt
    deployment_name = 'immerse-3-5'
    response = openai.Completion.create(
        engine=deployment_name,
        prompt=(f"Original Question: {prompt}\n"
                "Answer:"
                ),
        max_tokens=1000,
        n=1,
        temperature=0.3,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Extract CIDS from output
    output = response.choices[0].text.split('\n')[0]
    ids = re.findall("CID\d+", output)
    ids = list(set(ids))
    output = re.sub("\(?(CID\d+),?\)?|<\|im_end\|>|\[|\]", "", output)

    # Handle case where gpt doesn't output sources in prompt
    if ids is None or len(ids) == 0:
        alternates = ""
        for i in alts:
            alternates += f"{i.strip()}\n"
        return f"{output}\nBelow are some possible sources for reference", "**Confidence:** N/A"

    output = output[ 0 : output.rindex(".") + 1]
    confidence = utils.compute_average(ids,scores)
    conf_str = f"\n\n**Confidence:** {confidence:.2f}%"

    return output, conf_str

# Get responses
def get_responses(pipe, questions, answers, CIDs, source_links, source_filenames, SMEs, confidences, i):
    question = questions[i]

    output, conf, CIDs_i, source_links_i, source_filenames_i, SMEs_i, best_sme = get_response(pipe, question)
    CIDs_i, source_links_i, source_filenames_i, SMEs_i = utils.remove_duplicates(CIDs_i, source_links_i, source_filenames_i, SMEs_i)
    
    # Feed prompt into gpt, store query & output in session state
    answers[i] = output
    CIDs[i] = CIDs_i
    source_links[i] = source_links_i
    source_filenames[i] = source_filenames_i
    SMEs[i] = SMEs_i
    confidences[i] = conf

    print(f"Thread {threading.get_ident()} finished processing question {i+1}")

# Initiate gpt config data
def init_gpt():
    with open('gpt-config.json') as user_file:
        content = json.load(user_file)
    if content is None:
        raise Exception("Error reading config")

    openai.api_key = content["api_key"]
    openai.api_type = content["api_type"] 
    openai.api_version = content["api_version"]
    openai.api_base = content["api_base"]
