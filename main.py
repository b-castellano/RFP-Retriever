import pandas as pd
from datasets import load_dataset
import torch
import openai
import traceback

from haystack.nodes import EmbeddingRetriever
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers


import langchain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate


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
    return pipe.run(query=query, params={"Retriever": {"top_k": 3}})

# Create prompt template
def create_prompt(query, prediction):
    prompt = PromptTemplate(input_variables=["prefix", "question", "context"],
                            template="{prefix}\nQuestion: {question}\n Context: {context}\n")

    # Provide instructions/prefix
    prefix = """You are an assistant for the Information Security department of an enterprise designed to answer security questions 
    in a professional manner. Provided is the original question and some context consisting of a sequence of answers in the form of 
    'question ID, confidence score, and answer'. Use the answers within the context to formulate a concise response. In addition, list 
    the referenced question IDs of the answers you referenced at the end of your response."""

    # Create context
    context = ""
    for answer in prediction["answers"]:

        # Remove docs 
        context += "Question ID: {ID}, Content: {content}\n".format(
            ID=answer.meta["question ID"], content=answer.meta["answer"])
    # Generate Prompt
    print("Generating prompt...")
    print("PROMPT:", prompt.format(prefix=prefix, question=query, context=context))
    return prompt.format(prefix=prefix, question=query, context=context)
    


def init_gpt():

    openai.api_key = "dd9d2682f30f4f66b5a2d3f32fb6c917"
    openai.api_type = "azure"
    openai.api_version = "2023-05-15"
    openai.api_base = "https://immerse.openai.azure.com/"
    

# Call openai API
def call_gpt(prompt):

    deployment_name = 'immerse-3-5'
    response = openai.Completion.create(
        engine=deployment_name,
        prompt=(f"Question: {prompt}\n"
                "Answer:"
                ),
        max_tokens=500,
        n=1,
        top_p=0.7,
        temperature=0.3,
        frequency_penalty=0.5,
        presence_penalty=0.2
    )

    return response.choices[0].text.split('\n')[0]

def compute_average(used_docs):

    avgscore = 0
    count = 0

    for doc in used_docs:

        # Remove docs 
        avgscore += doc.score
        count+=1

    avgscore /= count   # convert total score to avg
    avgscore *= 100     # convert from decimal to percentage

    return avgscore


def main():

    try:
        # User's question
        query = "Has your organization implemented data loss prevention (DLP) to detect potential unauthorized access, use, or disclosure of client data?"

        # Initialize document store
        document_store, loaded = init_store()
    
        # Initialize retriever
        retriever = init_retriever(document_store)

        # If new store, add documents and embeddings
        if not loaded:
            write_docs(document_store, retriever)
        
        # Initialize pipeline for document search
        pipe = init_pipe(retriever)

        # Query database
        prediction = query_faiss(query, pipe)
        
        # Generate prompt from related docs
        prompt = create_prompt(query, prediction)

        # Initialize gpt-3
        init_gpt()

        # Feed prompt into gpt
        output = call_gpt(prompt)

        print(f"OUTPUT:\n======================={output}")

    except:
        
        print("Error initializing var")
        traceback.print_exc()

main()
