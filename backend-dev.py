import pandas as pd
from datasets import load_dataset
import traceback
import re
import json

from data_load import *

import openai

from haystack.nodes import EmbeddingRetriever
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers


import langchain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

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

def init_store():
    try:

        return FAISSDocumentStore.load(index_path="my_faiss_index.faiss"), True
        
    except:
        return FAISSDocumentStore(
            similarity="cosine",
            embedding_dim=768,
            duplicate_documents='overwrite'
        ), False

def init_retriever(document_store):
    return EmbeddingRetriever(
        document_store=document_store,
        embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
        model_format="sentence_transformers"
    )



def init_pipe(retriever):
    return FAQPipeline(retriever=retriever)

def init_gpt():

    with open('gpt-config.json') as user_file:
        content = json.load(user_file)

    if content is None:
        raise Exception("Error reading gpt-config")


    openai.api_key = content["api_key"]
    openai.api_type = content["api_type"] 
    openai.api_version = content["api_version"]
    openai.api_base = content["api_base"]

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

def get_response(pipe, query):
    
    prediction, closeMatch = query_faiss(query, pipe)

    # If a close match was found, just return that answer
    if closeMatch:
        newAnswer = re.sub("[\[\]'\"]","",prediction.meta["answer"])
        score = prediction.score * 100
        source = prediction.meta["cid"]
        
        return f"{newAnswer}\nConfidence Score: {score:.2f}%\nSources: \n{source}"
     
    # Generate prompt from related docs
    prompt,scores, alts = create_prompt(query, prediction)

    # Feed prompt into gpt
    return call_gpt(prompt, scores, alts)

def query_faiss(query, pipe):
    docs = pipe.run(query=query, params={"Retriever": {"top_k": 5}})

    # If there is a close match between user question and pulled doc, then just return that doc's answer
    print(docs["documents"])
    if docs["documents"][0].score > .95:
        return docs["documents"][0], True
        
    return docs, False


# Create prompt template
def create_prompt(query, prediction):

    prompt = PromptTemplate(input_variables=["prefix", "question", "context"],
                            template="{prefix}\nQuestion: {question}\n Context: ###{context}###\n")

    # Provide instructions/prefix
    prefix = """Assistant is a large language model designed to answer questions for an Information Security enterprise professionally. Provided is  some context consisting of a sequence of answers in the form of 'question ID, answer' and the question to be answered. Use the answers within the context to answer the question in a concise manner. At the end of your response, list the question IDs of the answers you referenced."""
    context = ""
    scores = {}
    alts = []
    count = 0

    
    for answer in prediction["answers"]:

        newAnswer = re.sub("[\[\]'\"]","",answer.meta["answer"])

        # Remove docs 
        context += "Question ID: {ID}, Answer: {answer}\n".format(
            ID=answer.meta["cid"], answer=newAnswer)
        
        # Add ID-Score pair to dict
        scores[answer.meta["cid"]] = answer.score

        if (count < 3):
            
            alts.append(answer.meta["cid"])
            count+=1

    input_prompt = prompt.format(prefix=prefix, question=query, context=context)

    messages=[
        {"role": "system", "content": input_prompt},
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

    ids = re.findall("CID\d+", output)
    ids = list(set(ids))
    output = re.sub("\(?(CID\d+),?\)?|<\|im_end\|>|\[(.*?)\]", "", output)


    # Handle case where gpt doesn't output sources in prompt
    if ids is None or len(ids) == 0:
        alternates = ""
        for i in alts:
            alternates += f"{i.strip()}\n"
        return f"{output}\nHere are some possible sources to reference:\n{alternates}"

    confidence = compute_average(ids,scores)
    output = output[ 0 : output.rindex(".") + 1]

    # If confidence is under 75%, output it cannot answer question
    if confidence < 75:
        alternates = ""
        for i in alts:
            alternates += f"{i.strip()}\n"
        return f"Sorry, I cannot answer that question.\nHere are some possible sources to reference:\n{alternates}"
   
    output += f"\nConfidence Score: {confidence:.2f}%"
    output += f"\nSources:\n"

    for i in ids:
        output += f"{i.strip()}\n"

    return output

def compute_average(ids, scores):

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
        pipe= init()

        while(True):
            
            # Users question
            query = input("Please ask a question. Reply 'STOP' to stop:")

            if query == "STOP":
                break

            output = get_response(pipe, query)
            print(f"OUTPUT:\n{output}")
        

       

    except:
        
        print("Error initializing var")
        traceback.print_exc()

if __name__ == "__main__":
    main()