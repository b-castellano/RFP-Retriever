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
    # Initialize document store
    document_store, loaded = init_store()

    # Initialize retriever
    retriever = init_retriever(document_store)

    # If new store, add documents and embeddings
    if not loaded:
        write_docs(document_store, retriever)
    
    # Initialize GPT
    init_gpt()

    # Initialize pipeline for document search
    return init_pipe(retriever)

# Initialize FAISS datastore
def init_store():
    try:
        return FAISSDocumentStore.load(index_path="my_faiss_index.faiss"), True
    except:
        return FAISSDocumentStore(
            similarity="cosine",
            embedding_dim=768,
            duplicate_documents='overwrite'
        ), False
    
# Initialize retriever for documents in vector DB
def init_retriever(document_store):
    return EmbeddingRetriever(
        document_store=document_store,
        embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
        model_format="sentence_transformers"
    )

# Initialize gpt configurations
def init_gpt():

    with open('gpt-config.json') as user_file:
        content = json.load(user_file)

    if content is None:
        raise Exception("Error reading gpt-config")


    openai.api_key = content["api_key"]
    openai.api_type = content["api_type"] 
    openai.api_version = content["api_version"]
    openai.api_base = content["api_base"]

# Initialize the pipeline
def init_pipe(retriever):
    return FAQPipeline(retriever=retriever)

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

# Get response for query
def get_response(pipe, query):
    prediction, closeMatch = query_faiss(query, pipe)

    # If a close match was found, just return that answer --> Clean up?
    if closeMatch:
        answer = re.sub("[\[\]'\"]","",prediction.meta["answer"])
        conf = prediction.score * 100
        cid = prediction.meta["cid"]
        source_link = prediction.meta["url"]
        source_filename = prediction.meta["file name"]
        sme = prediction.meta["sme"]
        best_sme = prediction.meta["sme"]
        
        return answer, conf, cid, source_link, sme, best_sme

    # No close match, so generate prompt from related docs
    else:
    
        messages, docs = create_prompt(query, prediction)
        answer, ids = call_gpt(messages)

        return answer, get_info(prediction, docs, ids)
        
def get_info(prediction, docs, ids):

    CIDs, SMEs, source_filenames, source_links, docs_used = []
    
    if ids == None: # If gpt did not find ids
        for answer in prediction:
            CIDs.append(answer.meta["cid"])
            source_links.append(answer.meta["url"])
            SMEs.append(answer.meta["sme"])
            source_filenames.append(answer.meta["file name"])
            docs_used[answer.meta["cid"]] = answer
    else:
        for id in ids:  # If gpt found ids
            CIDs.append(docs[id].meta["cid"])
            source_links.append(docs[id].meta["url"])
            SMEs.append(docs[id].meta["sme"])
            source_filenames.append(docs[id].meta["file name"])
            docs_used[docs[id].meta["cid"]] = answer

    conf = utils.compute_average_score(docs_used)

    return conf, CIDs, source_links, source_filenames, SMEs

        
# Get top k documents related to query from vector datastore
def query_faiss(query, pipe):
    docs = pipe.run(query=query, params={"Retriever": {"top_k": 5}})

    # If there is a close match (>=95% confidence) between user question and pulled doc, then just return that doc's answer
    print(docs["documents"])
    if docs["documents"][0].score >= .95:
        return docs["documents"][0], True
    
    # No close match
    else:
        return docs, False
        
# Create prompt template
def create_prompt(query, prediction):  ## May add a parameter "Short", "Yes/No", "Elaborate", etc. for answer preferences
    print("Creating prompt")
    prompt = PromptTemplate(input_variables=["prefix", "question", "context"],
                            template="{prefix}\nQuestion: {question}\n Context: ###{context}###\n")

    # Provide instructions/prefix
    prefix = """You are an assistant for the Information Security department of an enterprise designed to answer security questions professionally. Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, answer'. Use the answers within the context to answer the original question in a concise manner with explanation. Just at the end, list the question IDs of the answers you referenced to formulate your response."""
    
    # Create context
    context = ""

    # Used to get scores of prompts later on
    docs = {}

    for answer in prediction["answers"]:

        newAnswer = re.sub("[\[\]'\"]","",answer.meta["answer"])

        # Remove docs 
        context += "Question ID: {ID}, Answer: {answer}\n".format(
            ID=answer.meta["cid"], answer=newAnswer)
        
        # Add ID-Score pair to dict
        docs[answer.meta["cid"]] = answer
        
    system_prompt = prompt.format(prefix=prefix, question=query, context=context)

    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Is company information backed up regularly?"},
        {"role": "assistant", "content": 
        """
        Yes. (CID10491, CID94823)
        """},
        {"role": "user", "content": "How are offsite backup processes managed to protect against ransomware?"},
        {"role": "assistant", "content": 
        """
        All of our controls, monitoring, and policies are designed to prevent the spread of Ransomware anywhere in our 
        environment. Systems/Servers are backed up at a given site, those backups are then replicated to an UHG owned 
        offsite location over our private internal network. The physical servers, application and virtual servers being 
        backed up cannot access or modify the offsite copy of the data and are not accessible via the public internet.
        In other words if the original server gets infected, it cannot access the replicated copy, which is used for 
        restoration purposes only. The backup solution from IBM additionally provides ransomware detection configurations 
        which we have implemented to ensure that security alerts for potential data attacks are bubbled up for action at 
        the point of detection. (CID83724, CID53133, CID00947)
        """
        },
        {"role": "user", "content": query},
    ]

    return messages, docs
    
# Call openai API and compute confidence
def call_gpt(messages, docs):

    deployment_id = "deployment-ae1a29d047eb4619a2b64fb755ae468f"

    response = openai.ChatCompletion.create(
        engine=deployment_id,
        messages=messages,
        max_tokens=500,
        n=1,
        top_p=0.7,
        temperature=0.3,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    output = response['choices'][0]['message']['content']

    # Extract CIDs from gpt output
    ids = re.findall("CID\d+", output)

    ids = list(set(ids))
    output = re.sub("\(?(CID\d+),?\)?|<\|im_end\|>|\[(.*?)\]", "", output)

    # Handle case where gpt doesn't output sources in prompt
    if ids is None or len(ids) == 0:
        return output, None

    confidence = utils.compute_average_score(ids, docs)

    # If confidence is under 75%, output it cannot answer question
    if confidence < 75:
        return f"Sorry, I cannot answer that question.\nHere are some possible sources to reference:", None

    return output, ids
