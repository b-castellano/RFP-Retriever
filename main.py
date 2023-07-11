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
    return pipe.run(query=query, params={"Retriever": {"top_k": 4}})

# Create prompt template
def create_prompt(query, prediction, user_preference):
    ### user_preference might eventually become "Short", "Yes/No", "Elaborate", etc. for answer preferences
    print("Creating prompt")
    prompt = PromptTemplate(input_variables=["prefix", "question", "context"],
                            template="{prefix}\nQuestion: {question}\n Context: {context}\n")

    # Provide instructions/prefix
    ## TODO: create prefix variants based on user_preference value
    prefix = """You are an assistant for the Information Security department of an enterprise designed to answer security questions in a professional manner. Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, answer'. Use the answers within the context to formulate a response. In addition, list the question IDs of the answers you referenced at the end of your response in this form: [...,...]"""


    # Create context
    context = ""
    scores = {}
    SMEs, sme_dict, source_filenames, source_links, CIDs = [], {}, [], [], []  # SMEs are dictionary in this function to allow us to more easily get best_sme later
    for answer in prediction["answers"]:
        newAnswer = re.sub("[\[\]'\"]","",answer.meta["answer"])
        # Remove docs 
        context += "Question ID: {ID}, Answer: {answer}\n".format(
            ID=answer.meta["cid"], answer=newAnswer)
        
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
        if answer.meta["file name"] not in [None, ""]:
            source_filenames.append(answer.meta["file name"])
        else:
            source_filenames.append("N/A")

    # print("Scores:\n", scores)
    # Get the question ID with the highest score, use as primary SME
    max_id = max(scores, key=scores.get)
    best_sme = sme_dict[max_id]
    return prompt.format(prefix=prefix, question=query, context=context), scores, CIDs, source_links, source_filenames, SMEs, best_sme 
    

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
        temperature=0.1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    output = response.choices[0].text.split('\n')[0]

  
    ids = re.findall("CID\d+", output)
    output = re.sub("CID\d+", "", output)
    if ids is None:
        raise Exception("Error getting CID's")

    confidence = compute_average(ids,scores)

    # output = output[ 0 : output.index("[")]
    output = output[ 0 : output.rindex(".") + 1]

    sources = f"\n**Sources:**\n"
    for i in ids:
        sources += f"{i.strip()}\n"

    conf = f"\n\n**Confidence Score:** {confidence:.2f}%"

    return output, conf, sources



def compute_average(ids, scores):
    total = 0
    for id in ids:
        id = id.strip()
        total += scores[id]
    if len(ids) > 0:
        avgscore = total / len(ids)     # convert total score to avg
    else:
        avgscore = 0
    avgscore *= 100                 # convert from decimal to percentage
    return avgscore

def parse_sme_name(sme):
    # Split name, remove whitespace
    name_list = sme.replace('/', ',').split(',')
    name_list = [name.strip() for name in name_list]
    # Reorder
    if len(name_list) > 2:  # handle multiple names case
        firstnames = [name_list[i] for i in range(1, len(name_list))]
        lastname = name_list[0]
        fullname = ' '.join(firstnames) + ' ' + lastname
    else:  # handle single name case
        fullname = name_list[1] + ' ' + name_list[0]
    return fullname

### Setup session storage
st.session_state.responses = []

# Sidebar contents
with st.sidebar:
    st.title('RFP Retriever')
    st.image('./retrieverLogo.jpg')
    st.markdown('''
    ### About
    ChatBot to handle questions related to RFPs, EIS Artifacts, & SMEs
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

    # Init UI Slots
    st.header("Ask a Question:")
    options = np.array(["Short", "Regular", "Elaborate", "Yes/No"])
    # selected_option = st.selectbox('Desired answer type:', options=options, index=0)
    selected_option = 'TODO change this'

    query = st.text_input("RFP/Security-Related")
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

    if query: # If user submits a question
        # Query database, generate prompt from related docs, initialize gpt-3
        prediction = query_faiss(query, pipe)
        prompt, scores, CIDs, source_links, source_filenames, SMEs, best_sme = create_prompt(query, prediction, selected_option)
        init_gpt()
        # Feed prompt into gpt, store query & output in session state
        output, conf, sources = call_gpt(prompt, scores)
        st.session_state.responses.append(query)
        st.session_state.responses.append(output)

        ## Format output better for UI (awful hack):
        new_output = ""
        for i in range(0, len(output), 82):
            new_output += output[i:i+82] + "\n"

        response_header_slot.markdown(f"**Answer:**\n")
        response_slot.write(new_output)  
        with response_copy.expander('Copy response'):
            st.write("Copied response!")
            # pc.copy(output) ### COPY OUTPUT, HAS NO ADDED NEWLINES

        # Display confidence, sources, SMEs
        confidence_slot.markdown(conf)
        sources_header.markdown(f"**Sources:**\n")
        # create a markdown table
        markdown_table = "| CID | SME | File Name |\n| --- | --- | --- |\n|"
        for i in range(len(CIDs)):
            markdown_table += "[{0}]({1}) | {2} | {3} |\n|".format(CIDs[i], source_links[i], SMEs[i], source_filenames[i])
        sources_slot.write(markdown_table, unsafe_allow_html=True)
        best_sme = parse_sme_name(best_sme)
        best_sme_slot.markdown(f"**SME:** {best_sme} ")
        # Draft email option
        with draft_email.expander('Draft an email to the SME'):
            emailHeader.markdown("### Email To SME:")
            init_gpt()
            prompt = f"Please write a brief and professional business email to someone named {best_sme} asking {query}. Include only the text of the email in your response, not any sort of email address, and should be formatted nicely. The email should end with the exact string '[Your Name]'."
            response = openai.Completion.create(
                engine='immerse-3-5',
                prompt=prompt,
                temperature=0.3,
                max_tokens=100,
                frequency_penalty=0.0,
                presence_penalty=0,
            )
            email_response = response.choices[0].text
            res1 = email_response.find("[Your Name]")
            if res1 != -1:
                email_response = email_response[:res1+11]
            emailContent.text(email_response)
            print(email_response)
        # print(st.session_state.responses)

if __name__ == "__main__": 
    main()

