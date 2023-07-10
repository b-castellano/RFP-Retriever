# General
import warnings
from tqdm.auto import tqdm  # progress bar
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
import openai
import traceback
import os
import random
import re
from init_gpt import init_gpt
# import pyperclip as pc
# pc.copy("testing")
# x = pc.paste()
# print(x)

# Streamlit
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

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
    return pipe.run(query=query, params={"Retriever": {"top_k": 3}})

# Create prompt template
def create_prompt(query, prediction, user_preference):
    print("Creating prompt")
    prompt = PromptTemplate(input_variables=["prefix", "question", "context"],
                            template="{prefix}\nQuestion: {question}\n Context: {context}\n")

    # Provide instructions/prefix
    ## TODO: create prefix variants based on user_preference value
    prefix = """You are an assistant for the Information Security department of an enterprise designed to answer security questions in a professional manner. Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, answer'. Use the answers within the context to formulate a response. In addition, list the question IDs of the answers you referenced at the end of your response in this form: [...,...]"""
    # if user_preference == "Short":
    # elif user_preference == "Regular":
    # elif user_preference == "Elaborate":
    # else: # currently Yes/No Case

    # Create context
    context = ""
    scores = {}
    SMEs, file_names, CIDs = {}, [], []
    print("\n\n\n")
    print(prediction["answers"])
    print("\n\n\n")
    for answer in prediction["answers"]:
        newAnswer = re.sub("[\[\]'\"]","",answer.meta["answer"])
        # Remove docs 
        context += "Question ID: {ID}, Answer: {answer}\n".format(
            ID=answer.meta["question ID"], answer=newAnswer)
        
        # Add ID-Score pair to dict
        scores[answer.meta["question ID"]] = answer.score
        SMEs[answer.meta["question ID"]] = answer.meta["sme"]
        file_names.append(answer.meta["file name"])
        # CIDs.append(answer.meta["cid"])

    # Get the question ID with the highest score, use as primary SME
    max_id = max(scores, key=scores.get)
    best_sme = SMEs[max_id]

    # Generate Prompt
    # print("Generating prompt...")
    # print("PROMPT:", prompt.format(prefix=prefix, question=query, context=context))
    
    return prompt.format(prefix=prefix, question=query, context=context), scores, file_names, SMEs, best_sme
    

# Call openai API
def call_gpt(prompt,scores):
    print("Calling GPT")
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
  
    res = re.search("\[(.*)\]", output)
    if res is None:
        raise Exception("Error getting QID's")
    ids = re.split(",", res.group(1))

    confidence = compute_average(ids,scores)

    output = output[ 0 : output.index("[")]

    sources = f"\n**Sources:**\n"
    for i in ids:
        sources += f"{i.strip()}\n"

    conf = f"\n**Confidence Score:** {confidence:.2f}%"

    return output, conf, sources

def compute_average(ids, scores):
    print("Computing average")
    total = 0
    print(scores)
    print(ids)
    for id in ids:
        id = id.strip()
        total += scores[id]
    avgscore = total / len(ids)     # convert total score to avg
    avgscore *= 100                 # convert from decimal to percentage
    return avgscore
    

### User Interface

# Sidebar contents
with st.sidebar:
    st.title('RFP Retriever')
    st.image('./retrieverLogo.jpg')
    st.markdown('''
    ### About
    ChatBot to handle questions related to RFPs, Artifacts, & SMEs
    ''')
    add_vertical_space(5)
    st.write('By: *The Security Sages*')


def main():
    # Initialize document store
    document_store, loaded = init_store()
    # Initialize retriever
    retriever = init_retriever(document_store)
    # If new store, add documents and embeddings
    if not loaded:
        write_docs(document_store, retriever)
    # Initialize pipeline for document search
    pipe = init_pipe(retriever)
    # Initialize gpt bot
    init_gpt()

    st.header("RFP Retriever")

    # upload file option
    # file = st.file_uploader("Upload file(s) to be added to this query", type=['pdf', 'xls', 'xlsx', 'docx', 'csv'])
    options = np.array(["Short", "Regular", "Elaborate", "Yes/No"])
    # selected_option = st.selectbox('Desired answer type:', options=options, index=0)
    selected_option = 'TODO change this'

    query = st.text_input("Ask a question:")
    response_header_slot = st.empty()
    response_slot = st.empty()
    response_copy = st.empty()

    confidence_slot = st.empty()
    sources_header = st.empty()
    sources_slot = st.empty()
    best_sme_slot = st.empty()

    draft_email = st.empty()
    emailHeader = st.empty()
    emailContent = st.empty()

    if query:
        print("Calling query")
        # Query database
        prediction = query_faiss(query, pipe)
        
        # Generate prompt from related docs
        prompt,scores, source_filenames, SMEs, best_sme = create_prompt(query, prediction, selected_option)
        # Initialize gpt-3
        init_gpt()
        # Feed prompt into gpt
        output, conf, sources = call_gpt(prompt, scores)
        ## Format output better for UI (awful hack):
        new_output = ""
        for i in range(0, len(output), 82):
            new_output += output[i:i+82] + "\n"

        response_header_slot.markdown(f"**Answer:**\n")
        response_slot.text(new_output)
        with response_copy.expander('Copy response'):
            st.write("Copied response!")
            # pyCopy(output) ### COPY OUTPUT, HAS NO ADDED NEWLINES

        confidence_slot.markdown(conf)
        sources_table = "\n".join([f"{source}" for source in source_filenames])
        sources_string = f"**Sources:**\n"
        sources_header.markdown(sources_string)
        sources_slot.text(sources_table)
        # sources_slot.table(sources) # TODO fix here

        best_sme_slot.markdown(f"**SME:** {best_sme} ")

        with draft_email.expander('Draft an email to the SME'):
            emailHeader.markdown("### Email To SME:")
            init_gpt()
            prompt = f"Please write a brief and professional business email to {best_sme} asking {query}. Include only the email in your response, and format it nicely"
            response = openai.Completion.create(
                engine='immerse-3-5',
                prompt=prompt,
                temperature=0,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            print(response.choices)
            email_response = response.choices[0]
            print(type(email_response))
            res1 = email_response.index("Question 2")
            if res1 != -1:
                email_response = email_response[:res1]
            emailContent.text(email_response)

if __name__ == "__main__": 
    main()

