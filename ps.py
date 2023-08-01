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

from func_timeout import func_timeout, FunctionTimedOut
from response import Response

import app

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
def get_responses(pipe, questions, answers, CIDs, source_links, best_SMEs, confidences, i, lock, num_complete, progress_text, progress_bar):
    print(f"Running question {i + 1}")
    question = questions[i]
    response = Response()

    # Get relavent response for question
    response = get_response(pipe, question, lock)

    # Check if source links and source filenames are not lists
    source_links_i = response.source_links

    if type(source_links_i) == str:
       source_links_i = [source_links_i]


    # source_links_i = list(filter(None, source_links_i))
    if source_links_i is None:
        source_links_i = [["N/A"]]
   

    # Filter out None entries in lists
    #source_links_i = list(filter(None, source_links_i))
    #source_filenames_i = list(filter(None, source_filenames_i))

    # Feed prompt into gpt, store query & output in session state for threads
    lock.acquire()
    answers[i] = response.answer
    cids[i] = response.cids
    source_links[i] = source_links_i
    best_smes[i] = response.best_sme
    confidences[i] = response.conf
    lock.release()

    print(f"Thread {threading.get_ident()} finished processing question {i+1}")
    lock.acquire()
    num_complete.append(num_complete.pop() + 1)
    print("num_complete:", num_complete[0])
    lock.release()
    # progress_text = "Questions being answered, please wait."
    # progress_bar = st.progress((num_complete[0] / len(questions)), text=progress_text)
    progress_bar.progress((num_complete[0] / len(questions)), progress_text)

# Get response for query
def get_response(pipe, query, lock=threading.Lock()):
    lock.acquire()

    prediction, closeMatch = query_faiss(query, pipe)
    response = Response()

    # If a close match was found, just return that answer
    if closeMatch:
        answer = prediction.meta["answer"].split(",")
        response.answer = simplify_answer(query, re.sub("[\[\]'\"]","", answer[0]))
        response.conf = f"{round((prediction.score * 100),2)}%"
        response.cids = [prediction.meta["cid"]]
        response.source_links = [prediction.meta["url"]]
        response.smes = [prediction.meta["sme"]]
        response.best_sme = prediction.meta["sme"]
        lock.release()
        
        return response

    # No close match, so generate prompt from related docs
    else:
    
        messages, docs = create_prompt(query, prediction)
        lock.release()
        try:
            foo = "foo"
            answer, ids = func_timeout(15, call_gpt, args=(messages, foo))
        except FunctionTimedOut:
            print("Restarting GPT call")
            return get_response(pipe, query, lock)

        response = get_info(prediction, docs, ids)
        response.answer = simplify_answer(query, answer)
        
        # If confidence is under 75%, output it cannot answer question --> Disabled for debug purposes
        #if conf < 75:
        #    answer = "Sorry, I cannot answer that question.\nHere are some possible sources to reference:"

        return response


# Get top k documents related to query from vector datastore
def query_faiss(query, pipe):
    docs = pipe.run(query=query, params={"Retriever": {"top_k": 5}})

    # If there is a close match (>=95% confidence) between user question and pulled doc, then just return that doc's answer
    if docs["documents"][0].score >= .95:
        return docs["documents"][0], True
    
    # No close match
    else:
        return docs, False
        
# Create prompt template
def create_prompt(query, prediction):

    print("Creating prompt")
    prompt = PromptTemplate(input_variables=["prefix", "question", "context"],
                            template="{prefix}\nQuestion: {question}\n Context: ###{context}###\n")

    # Provide instructions/prefix
    # prefix = """You are an assistant for the Information Security department of an enterprise designed to answer security questions professionally. 
    # Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, answer'. Use the answers 
    # within the context to answer the original question in a concise manner. If the question can be answered with yes or no, then do so. Just at the end, 
    # list the question IDs of the answers you referenced to formulate your response."""
    
    prefix = """Assistant is a large language model designed by the Security Sages to answer questions for an Information Security enterprise professionally. 
    Provided is some context consisting of a sequence of answers in the form of 'question ID, answer' and the question to be answered. 
    Use the answers within the context to answer the question in a concise manner. At the end of your response, list the question IDs of the answers you referenced."""

    context = ""
    docs = {} # Used to get scores of prompts later on
    for answer in prediction["answers"]:
        newAnswer = re.sub(r"[\[\]'\"]","",answer.meta["answer"])

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
        Yes. (CID46487)
        """},
        {"role": "user", "content": "Is your Business Continuity Management program certified under any frameworks?"},
        {"role": "assistant", "content": 
        """
        No. (CID46888)
        """},
        {"role": "user", "content": "Does your company provide Information Security Training?"},
        {"role": "assistant", "content": 
        """
        Yes. (CID46476)
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
        {"role": "user", "content": "For Live and Work Well, what Cloud Service Providers does your solution support?"},
        {"role": "assistant", "content": 
        """
        For externally hosted applications, MS Azure is a preferred Cloud Service Provider for Optum Behavioral Health. (CID55595)
        """
        },
        {"role": "user", "content": query},
    ]

    return messages, docs
    
# Call openai API and compute confidence
def call_gpt(messages, foo):

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
    print(output)

    # Extract cids from gpt output
    ids = re.findall(r"CID\d+", output)

    # Clean up cids
    ids = list(set(ids))
    output = re.sub(r"\(?(CID\d+),?\)?|<\|im_end\|>|\[(.*?)\]", "", output)

    # Handle case where gpt doesn't output sources in prompt
    if ids is None or len(ids) == 0:
        return output, None

    return output, ids

# Gets additional data for output
def get_info(prediction, docs, ids):
    response = Response()
    cids, smes, source_links = [], [], []
    docs_used = {}

    if ids == None: # If gpt did not find ids
        ids = []

        # Use all ids
        for answer in prediction["answers"]:
            ids.append(answer.meta["cid"])
        
    
    ids = list(set(ids)) ## Remove duplicates in found ids
    best_score = 0

    for id in ids:  ## If gpt found ids
        
        # Get relavent data for returned ids
        cids.append(docs[id].meta["cid"])
        source_links.append(docs[id].meta["url"])
        smes.append(docs[id].meta["sme"])
        docs_used[docs[id].meta["cid"]] = docs[id]

        # Find sme with highest confidence document
        if best_score < docs_used[id].score:
            best_sme = docs_used[id].meta["sme"]

    # Get average confidence score for used documents
    conf = utils.compute_average_score(docs_used)
    conf = f"{round(conf,2)}%"
    
    # Populate response object with info
    response.conf = conf
    response.best_sme = best_sme
    response.cids = cids
    response.source_links = source_links
    response.smes = smes

    return response

    # Searches answer for yes or no response and outputs that for simplified answer
def simplify_answer(query, answer):
    query = query.strip()
    answer = answer.strip()

    if (query[len(query) -1] != "?"):
        return answer

    else:
        firstWord = answer.split(" ")[0]
        if re.search(r"([Yy]es)", firstWord) is not None:
            return "Yes."
        elif re.search(r"([Nn]o)", firstWord) is not None:
            return "No."
        else:
            return answer
