from datasets import load_dataset
import traceback
import re
import json

import openai

from haystack.nodes import EmbeddingRetriever
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers
from haystack.pipelines import DocumentSearchPipeline
from haystack.nodes import FARMReader

import sharepoint.sp_load as sp_load
import rfpio.rfp_load as rfp_load

import langchain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

### Helper functions, generally pertaining to queries themselves (back-end)


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

def get_response(rfp_pipe, sp_pipe, query):
    
    rfp_prediction, sp_prediction, closeMatch = query_faiss(query, rfp_pipe,sp_pipe)

    # If a close match was found (direct RFPIO document), just output that answer
    if closeMatch:

        if rfp_prediction is None:
            prediction = sp_prediction
        else:
            prediction = rfp_prediction

        answer = simplify_answer(query, re.sub(r"[\[\]'\"]","",prediction.meta["answer"]))

        score = prediction.score * 100
        source = prediction.meta["cid"]
        
        
        return f"{answer}\nConfidence Score: {score:.2f}%\nSources: \n{source}"

    # No close match, so generate prompt from related docs (call GPT)
    else:
        prompt,scores, alts = create_prompt(query, rfp_prediction, sp_prediction)

        # Feed prompt into gpt
        answer, score, sources = call_gpt(prompt, scores, alts)

        answer = simplify_answer(query, answer)

        return f"{answer}\nConfidence Score: {score:.2f}%\nSources: \n{sources}"



# Get top k documents related to query from vector datastore
def query_faiss(query, rfp_pipe, sp_pipe):
    rfp_docs = rfp_pipe.run(query=query, params={"Retriever": {"top_k": 5}})
    sp_docs = sp_pipe.run(query=query, params={"Retriever": {"top_k": 5}})

    for doc in sp_docs["documents"]:
        print(f"=====================\n{doc.content}, {doc.score}, {doc.meta['filename']}\n==================")
        
    # If there is a close match (>=95% confidence) between user question and pulled doc, then just return that doc's answer
    if rfp_docs["documents"][0].score >= .95:
        return rfp_docs["documents"][0],None, True

    # No close match, use all rfp and sharepoint docs
    else:

        return rfp_docs, sp_docs, False


# Create prompt template
def create_prompt(query, rfp_prediction, sp_prediction):

    prompt = PromptTemplate(input_variables=["prefix", "question", "context"],
                            template="{prefix}\nQuestion: {question}\n Context: ###{context}###\n")

    # Provide instructions/prefix
    prefix = """Assistant is a large language model designed by the Security Sages to answer questions for an Information Security enterprise professionally. 
    Provided is some context consisting of a sequence of answers in the form of 'question ID, answer' and the question to be answered. 
    Use the answers within the context to answer the question in a concise manner. At the end of your response, list the question IDs of the answers you referenced."""

    context = ""
    scores = {}
    alts = []
    count = 0

    
    for answer in rfp_prediction["answers"]:

        newAnswer = re.sub(r"[\[\]'\"]","",answer.meta["answer"])

        # Remove docs 
        context += "Question ID: {ID}, Answer: {answer}\n".format(
            ID=answer.meta["cid"], answer=newAnswer)
        
        # Add ID-Score pair to dict
        scores[answer.meta["cid"]] = answer.score

        if (count < 3):
            
            alts.append(answer.meta["cid"])
            count+=1

    # for answer in sp_prediction["documents"]:

    #     filename = answer.meta["filename"]
    #     page = answer.meta["page"]
    #     file_info = f"{filename}, Page {page}"

    #     context += "Question ID: {info}, Answer: {answer}\n".format(
    #         info=file_info, answer=answer.answer)

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
    return messages, scores, alts

# Call openai API
def call_gpt(messages,scores,alts):

    deployment_id = "deployment-ae1a29d047eb4619a2b64fb755ae468f"

    response = openai.ChatCompletion.create(
        engine=deployment_id,
        messages=messages,
        max_tokens=500,
        n=1,
        top_p=0.7,
        temperature=0.3
        ,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    output = response['choices'][0]['message']['content']

    # Extract CIDs from gpt output
    ids = re.findall(r"CID\d+", output)

    ids = list(set(ids))
    output = re.sub(r"\(?(CID\d+),?\)?|<\|im_end\|>|\[(.*?)\]", "", output)


    # Handle case where gpt doesn't output sources in prompt
    if ids is None or len(ids) == 0:
        alternateSources = ""
        for i in alts:
            alternateSources += f"{i.strip()}\n"
        return f"{output}\nHere are some possible sources to reference:\n", 0, alternateSources

    confidence = compute_average(ids,scores)

    # If confidence is under 75%, output it cannot answer question
    if confidence < 75:
        alternates = ""
        for i in alts:
            alternates += f"{i.strip()}\n"
        return f"Sorry, I cannot answer that question.\nHere are some possible sources to reference:\n", 0, alternates
   
    score = confidence

    sources = ""

    for i in ids:
        sources += f"{i.strip()}\n"

    return output, score, sources

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


def compute_average(ids, scores):
    print(ids)
    print(scores)
    total = 0

    for id in ids:

        id = id.strip()
        total += scores[id]

    avgscore = total / len(ids)     # convert total score to avg
    avgscore *= 100                 # convert from decimal to percentage

    return avgscore



def main():

    try:

        # Initialize FAISS store and create pipe instance
        rfp_pipe, sp_pipe = init()

        while(True):
            
            # Users question
            query = input("Please ask a question. Reply 'STOP' to stop:")

            if query == "STOP":
                break

            output = get_response(rfp_pipe, sp_pipe, query)
            print(f"OUTPUT:\n{output}")
        

    except:
        
        print("Error initializing var")
        traceback.print_exc()

if __name__ == "__main__":
    main()