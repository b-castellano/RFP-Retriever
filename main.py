# General
import warnings
import os
import pandas as pd
from datasets import load_dataset
import torch
import openai
import traceback
import re

# Haystack
from haystack.nodes import EmbeddingRetriever
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers

# Langchain
import langchain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

# Warning filter
warnings.filterwarnings('ignore', "TypedStorage is deprecated", UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def init_pipe(retriever):
    return FAQPipeline(retriever=retriever)

def query_faiss(query, pipe):
    # while True:
    # query = input("What question would you like to ask? (Type \"STOP\" to exit): ")
    # if query == "STOP":
    #     break
    return pipe.run(query=query, params={"Retriever": {"top_k": 5}})

# Create prompt template
def create_prompt(query, prediction):
    if "?" not in query:
        query += "?"

    template = """"You are an assistant for the Information Security department of an enterprise designed to answer security questions in a professional manner.
    Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, answer'.
    Use the answers within the context to formulate a concise response.
    List the question IDs of the answers you referenced to formulate your response."
    
    Question: {question}
    
    Context: {context}
    """
    gpt_template = PromptTemplate (
        input_variables=["question", "context"],
        template=template
    )

    # Create context
    context = ""
    scores = {}
    for answer in prediction["answers"]:
        prompt_context = re.sub("[\[\]'\"]","",answer.meta["answer"])
        # Remove docs 
        context += "Question ID: {ID}, Answer: {answer}\n".format(ID=answer.meta["cid"], answer=prompt_context)
        
        # Add ID-Score pair to dict
        scores[answer.meta["cid"]] = answer.score

    # Generate Prompt
    print("Generating prompt...")
    print("PROMPT:", gpt_template.format(question=query, context=context))
    
    return gpt_template.format(question=query, context=context), scores

def init_gpt():
    openai.api_key = "dd9d2682f30f4f66b5a2d3f32fb6c917"
    openai.api_type = "azure"
    openai.api_version = "2023-06-01-preview"
    openai.api_base = "https://immerse.openai.azure.com/"
    
# Call openai API
def call_gpt(prompt,scores):
    deployment_name = 'immerse-3-5'
    response = openai.Completion.create(
        engine=deployment_name,
        prompt=(prompt),
        max_tokens=500,
        n=1,
        top_p=0.7,
        temperature=0.3,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    output = response.choices[0].text.split('\n')[0]
    print(output)
    
    ids = re.findall("CID\d+", output)
    output = re.sub("CID\d+", "", output)

    if ids is None or len(ids) == 0:
        raise Exception("Error getting CID's")

    confidence = compute_average(ids,scores)
    output = output[ 0 : output.rindex(".") + 1]
   
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
    
    avgscore = total / len(ids)
    avgscore *= 100 

    return avgscore

def main():
    try:
        # Initialize document store
        document_store, loaded = init_store()

        # Initialize retriever
        retriever = init_retriever(document_store)
        
        # Check if docs are stored
        if not loaded:
            write_docs(document_store, retriever)

        # Initialize pipeline for document search
        pipe = init_pipe(retriever)

        # Initialize gpt bot
        init_gpt()

        while True:
            #query = input("What question would you like to ask? (Type \"STOP\" to exit): ")
            #if query == "STOP":
            #    break
            #good_query = "Please describe how you secure data at rest."
            #bad_query = "Are encryption keys managed and maintained?"

            count = 10
            file = open("Output_4.txt", "w")
            for n in range(count):
                df = pd.read_csv("qna.csv")
                query = df["question"][n]
                print(query)

                # Query Database
                prediction = query_faiss(query, pipe)

                # Generate prompt from related docs
                prompt, scores = create_prompt(query, prediction)

                # Feed prompt into gpt
                output = call_gpt(prompt, scores)

                print(f"Question: {query}\n OUTPUT: {output}\n")

                txt = (f"Question: {query}\n" + f"{output}\n")
                file.write(txt)
            file.close()
            break
    except:
        print("Error initializing var")
        traceback.print_exc()

if __name__ == "__main__": 
    main()