# General
import warnings
from tqdm.auto import tqdm  # progress bar
from datasets import load_dataset
import pandas as pd
import torch
import openai
import traceback
import re

# Haystack
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack import Document
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers

# Langchain
import langchain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser


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
<<<<<<< HEAD

def init_pipe(retriever):
    return FAQPipeline(retriever=retriever)

def query_faiss(query, pipe):
    return pipe.run(query=query, params={"Retriever": {"top_k": 4}})
=======
>>>>>>> 92c64b38e979bc2f9ca24c99797645575042c2a0

def create_prompt(query, prediction):
    prompt = gpt_template_dylan
    IDs = {}

<<<<<<< HEAD
    # Create context
    prompt_context = ""
    prompt_ids = ""
    avgscore = 0
    count = 0
    for answer in prediction["answers"]:
        id = answer.meta["question ID"]
        prompt_context = re.sub("[\[\]']","",answer.meta["answer"])
        score = answer.score
        prompt_context += "Question ID: {ID}\n Content: {content}\n".format(ID=id, content=prompt_context)
        prompt_ids += "{ID}\n".format(ID=id)

        IDs[id] = score
        avgscore += answer.score
        count += 1
    avgscore /= count

    print("Generating prompt...")
    return prompt.format(question=query, context=prompt_context), IDs

def init_gpt():
    openai.api_key = "dd9d2682f30f4f66b5a2d3f32fb6c917"
    openai.api_type = "azure"
    openai.api_version = "2023-06-01-preview"
    openai.api_base = "https://immerse.openai.azure.com/"
    
def call_gpt(prompt):
    deployment_name = 'immerse-3-5'
    response = openai.Completion.create(
        engine=deployment_name,
        prompt=(prompt),
        max_tokens=1000,
        n=1,
        top_p=0.7,
        temperature=0.2,
        frequency_penalty=0.5,
        presence_penalty=0.2
    )
    return response.choices[0].text.split('\n')[0]

def compute_average(gpt, dict):
    avgscore = 0
    try:
        txt = re.search("\[(.*)\]", gpt)
        txt_split = re.split(",\s?", txt.group(1))
        count = 0
        for word in txt_split:
            avgscore += dict[word]
            count += 1
        avgscore /= count
    except:
        print("Cannot Find")
    return avgscore

def main():
    # Initialize document store
    loaded = False
    document_store, loaded = init_store()

    # Initialize retriever
    retriever = init_retriever(document_store)
    
    # Check if docs are stored
    if not loaded:
        write_docs(document_store, retriever)

    # Initialize pipeline
    pipe = init_pipe(retriever)

    # Initialize gpt bot
    init_gpt()

    while True:
        #query = input("What question would you like to ask? (Type \"STOP\" to exit): ")
        #if query == "STOP":
        #    break

        # good_query = "Please describe how you secure data at rest."
        # bad_query = "Are encryption keys managed and maintained?"
        count = 100
        file = open("Output_2.txt", "w")
        for n in range(count):
            df = pd.read_csv("qna.csv")
            query = df["question"][1]
            print(query)

            #query = "Are encryption keys managed and maintained?"

            # Get relavant answers from database
            prediction = query_faiss(query, pipe)

            # Construct the prompt and dictionary
            prompt, dict = create_prompt(query, prediction)

            # Get gpt output for prompt
            gpt_output = call_gpt(prompt)

            # Get avaerage confidence interval for relavant answers
            avgscore = compute_average(gpt_output, dict)

            # Calculate failure rate
            fail_count = 0
            if avgscore < 1:
                fail_count += 1

            print(f"Question: {query}\n Output: {gpt_output}\n Score: {avgscore}")

            txt = (f"Question: {query}\n" + f"{gpt_output}\n" + f"{avgscore}\n")
            file.write(txt)
        file.close()

        fail_count /= count
        print(f"Fail Ratio: {fail_count}")
        break

if __name__ == "__main__": 
=======
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
    prefix = """You are an assistant for the Information Security department of an enterprise designed to answer security questions in a professional manner. Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, answer'. Use the answers within the context to formulate a concise response. In addition, list the question IDs of the answers you referenced at the end of your response in this form: [..,..]"""

    # Create context
    context = ""
    scores = {}
    for answer in prediction["answers"]:
        newAnswer = re.sub("[\[\]'\"]","",answer.meta["answer"])
        # Remove docs 
        context += "Question ID: {ID}, Answer: {answer}\n".format(
            ID=answer.meta["question ID"], answer=newAnswer)
        
        # Add ID-Score pair to dict
        scores[answer.meta["question ID"]] = answer.score

    # Generate Prompt
    print("Generating prompt...")
    print("PROMPT:", prompt.format(prefix=prefix, question=query, context=context))
    
    return prompt.format(prefix=prefix, question=query, context=context), scores
    


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
    output = response.choices[0].text.split('\n')[0]
    # print (output)
  
    res = re.search("\[(.*)\]", output)
    if res is None:
        raise Exception("Error getting QID's")
    ids = re.split(",", res.group(1))
    
    confidence = compute_average(ids,scores)

    output = output[ 0 : output.index("[")]
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
        # User's question
        query = "Describe your disaster recovery program.  Do you have an offset storage and facility?"

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
        prompt,scores = create_prompt(query, prediction)

        # Initialize gpt-3
        init_gpt()

        # Feed prompt into gpt
        output = call_gpt(prompt, scores)

        print(f"OUTPUT:\n{output}")

    except:
        
        print("Error initializing var")
        traceback.print_exc()

if __name__ == "__main__":
>>>>>>> 92c64b38e979bc2f9ca24c99797645575042c2a0
    main()