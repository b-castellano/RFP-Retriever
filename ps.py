# General
from operator import index
import sys
import openai
import pandas as pd
import re
import threading
import json
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import FAQPipeline
from haystack.pipelines import DocumentSearchPipeline
from langchain.prompts import PromptTemplate
import utils
from func_timeout import func_timeout, FunctionTimedOut

import rfpio.rfp_load as rfp_load
import sharepoint.sp_load as sp_load

# External Files
from response import Response

### GPT & faiss calling back-end functions

def init():
    # Initialize document store
    rfp_store, rfp_loaded, sp_store, sp_loaded = init_stores()

    # Initialize rfp retriever
    rfp_retriever = init_retriever(rfp_store)

    # Initialize sp retriever
    sp_retriever = init_retriever(sp_store)

    # If new store, add documents and embeddings

    if not rfp_loaded:
        write_rfp_docs(rfp_store, rfp_retriever)

    
    if not sp_loaded:
        write_sp_docs(sp_store, sp_retriever)
    
    # Initialize GPT
    init_gpt()

    # Initialize pipeline for document search
    return init_pipes(rfp_retriever,sp_retriever)

# Initialize FAISS datastore
def init_stores():

    rfp_store, rfp_loaded = rfp_load.init_store()

    sp_store, sp_loaded = sp_load.init_store()

    return rfp_store, rfp_loaded, sp_store, sp_loaded

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
def init_pipes(rfp_retriever, sp_retriever):
    return FAQPipeline(retriever=rfp_retriever), DocumentSearchPipeline(retriever=sp_retriever)

# Add RFPIO data to VectorDB
def write_rfp_docs(rfp_document_store, retriever):
    print("Writing RFPIO Documents...")
    
    rfp_load.parseQNAandEmbedDocuments(rfp_document_store, retriever)


# Add SharePoint data to VectorDB
def write_sp_docs(sp_document_store, sp_retriever):
    print("Writing SharePoint Documents...")
    directory = r"/Users/dhoule5/OneDrive - UHG/EIS Artifacts/"

    filenames = sp_load.getAllFileNames(directory)

    sp_load.parseFilesAndEmbedDocuments(directory, filenames, sp_document_store, sp_retriever)

# Get responses
def get_responses(rfp_pipe,sp_pipe, questions, answers, cids, source_links, best_smes, confidences, i, lock, num_complete, progress_text, progress_bar):

    print(f"Running question {i + 1}")
    question = questions[i]
    response = Response()

    # Get relevant response for question
    response = get_response(rfp_pipe, sp_pipe, question, lock=lock)

    # Remove empty strings in lists
    source_links_i = response.source_links
    source_filenames_i = response.source_filenames
    source_links_i = list(filter(lambda x: x != '', source_links_i))
    if source_links_i is None:
        source_links_i = [["N/A"]]
    source_filenames_i = list(filter(lambda x: x != '', source_filenames_i))
    if source_filenames_i is None:
        source_filenames_i = [["N/A"]]

    # Output response to session state
    lock.acquire()
    answers[i] = response.answer
    cids[i] = response.cids
    source_links[i] = source_links_i
    best_smes[i] = response.best_sme
    confidences[i] = response.conf
    lock.release()

    # Update the number of completed threads/questions and move progress bar
    print(f"Thread {threading.get_ident()} finished processing question {i+1}")

    lock.acquire()
    num_complete.append(num_complete.pop() + 1)
    print("num_complete:", num_complete[0])
    lock.release()

    progress_text = f"Questions being answered, please wait. ({num_complete[0]} / {len(questions)} complete)"
    progress_bar.progress((num_complete[0] / len(questions)), progress_text)

# Get response for query
def get_response(rfp_pipe, sp_pipe, query, lock=threading.Lock(), history=["N/A"], retries=0):
    lock.acquire()

    rfp_prediction, sp_prediction, closeMatch = query_faiss(query, rfp_pipe,sp_pipe)

    response = Response()

    # If a close match was found to rfp doc, just return that answer
    if closeMatch:
        answer = rfp_prediction.meta["answer"]
        response.answer = simplify_answer(query, re.sub(r"[\[\]'\"]", "", answer))
        response.conf = f"{round((rfp_prediction.score * 100),2)}%"
        response.cids = [rfp_prediction.meta["cid"]]
        response.source_links = [rfp_prediction.meta["url"]]
        response.smes = [rfp_prediction.meta["sme"]]
        response.best_sme = rfp_prediction.meta["sme"]
        lock.release()
        
        return response

    # No close match, so generate prompt from related docs
    else:
        messages, docs = create_prompt(query, rfp_prediction, sp_prediction, history)
        lock.release()

        try:
            foo = "foo"
            answer, ids = func_timeout(10, call_gpt, args=(messages, foo))
        except FunctionTimedOut:
            if retries == 3:
                print(f"GPT call failed on the following question: {query}")
                response.answer = "GPT call failed"
                response.conf = "0%"
                response.cids = ["N/A"]
                response.source_links = ["N/A"]
                response.source_filenames = ["N/A"]
                response.smes = ["N/A"]
                response.best_sme = "N/A"
                return response

            print("Restarting GPT call")
            return get_response(rfp_pipe, sp_pipe, query, lock=lock, retries=(retries + 1))

        response = get_info(rfp_prediction, sp_prediction, docs, ids)
        response.answer = simplify_answer(query, answer)
        
        # If confidence is under 75%, output it cannot answer question --> Disabled for debug purposes
        #if conf < 75:
        #    answer = "Sorry, I cannot answer that question.\nHere are some possible sources to reference:"

        return response

# Get top k documents related to query from vector datastore
def query_faiss(query, rfp_pipe, sp_pipe):
    rfp_docs = rfp_pipe.run(query=query, params={"Retriever": {"top_k": 5}})
    sp_docs = sp_pipe.run(query=query, params={"Retriever": {"top_k": 5}})

    # If there is a close match (>=95% confidence) between user question and pulled rfp doc, return that doc's answer
    if rfp_docs["documents"][0].score >= .95:
        return rfp_docs["documents"][0],None, True

    # No close match, use all rfp and sharepoint docs
    else:
        return rfp_docs, sp_docs, False
        
# Create prompt template
def create_prompt(query, rfp_prediction, sp_prediction, history):
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
    Use the answers within the context to answer the question in a concise manner. At the end of your response, list the question IDs of the answers you referenced.
    If you can't answer the question just say 'I'm sorry I cannot answer that question.'"""

    context = ""
    docs = {} # Used to get scores of prompts later on
    for answer in rfp_prediction["answers"]:
        newAnswer = re.sub(r"[\[\]'\"]","",answer.meta["answer"])

        # Add doc to context
        context += "Question ID: {ID}, Answer: {answer}\n".format(
            ID=answer.meta["cid"], answer=newAnswer)
        
        # Add ID-Score pair to dict
        docs[answer.meta["cid"]] = answer


    for doc in sp_prediction["documents"]:
        answer = doc.content
        context += "Question ID: {ID}, Answer: {answer}\n".format(
            ID=doc.meta["filename"], answer=answer)

        
    system_prompt = prompt.format(prefix=prefix, question=query, context=context)
    print(system_prompt)
    # Few-shot training examples
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
        {"role": "user", "content": "What does Optum think of Apple?"},
        {"role": "assistant", "content": 
        """
        Sorry I cannot answer that question. (CID29004)
        """
        }
    ]

    if len(history) > 10:
        history = history[-10:]

    for pair in history:
        if type(pair) == list:
            messages.append({"role": "user", "content": pair["question"]})
            messages.append({"role": "assistant", "content": pair["answer"]})

    messages.append({"role": "user", "content": query})
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

    # Extract sources and remove duplicates
    ids = list(set(ids))
    output = re.sub(r"\(?(CID\d+),?\)?|<\|im_end\|>|\[(.*?)\]", "", output)
    

    return output, ids

# Gets additional data for output
def get_info(rfp_prediction, sp_prediction, docs, ids):
    response = Response()
    cids, smes, source_links = [], [], []
    docs_used = {}

    if ids == None or len(ids) == 0: # If gpt did not find ids
        ids = []

        # Use all ids to get information
        for answer in rfp_prediction["answers"]:
            ids.append(answer.meta["cid"])
        
    ids = list(set(ids)) ## Remove duplicates in found ids
    best_score = 0
    best_sme = "Not Found"

    for id in ids:  
        
        try: ## Check if a CID given by gpt is invalid (not real)
            docs[id]
        except: ## If so, skip it
            continue

        # Get relevant data for returned ids
        cids.append(docs[id].meta["cid"])
        source_links.append(docs[id].meta["url"])
        smes.append(docs[id].meta["sme"])
        docs_used[docs[id].meta["cid"]] = docs[id]

        # Find sme with highest confidence document
        if docs_used[id].score > best_score and docs_used[id].meta["sme"] != "":
            best_sme = docs_used[id].meta["sme"]

    # Get average confidence score for used documents
    if len(docs_used) == 0:
        conf = 0
    else:
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
    
    # Clean up questions and answer
    query = query.strip()
    answer = answer.strip()
    firstAnswerWord = answer.split(" ")[0]
    firstQueryWord = query.split(" ")[0]
  
    # If not a Yes/No question, provide the entire response with description
    if query[len(query) -1] != "?" and re.search(r"([Ii]s)|(Does)|(Do)", firstQueryWord) is None:
        return answer
    else:
        # If explanation is requested, provide entire response
        if re.search(r"([Ee]xplain)|([Dd]escribe)", query) is not None:
            return answer
        
        # Else, output Yes or No depending on response
        elif re.search(r"([Yy]es)", firstAnswerWord) is not None:
            return "Yes."
        elif re.search(r"([Nn]o)", firstAnswerWord) is not None:
            return "No."
        else:
            return answer