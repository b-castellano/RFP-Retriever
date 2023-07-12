# General
import warnings
# from tqdm.auto import tqdm  # progress bar
# from datasets import load_dataset
import numpy as np
import pandas as pd
import openai
import traceback
import os
import re
import json
import pyperclip as pc
import utils

# Streamlit
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

# Haystack
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import FAQPipeline

# Langchain
import langchain
from langchain.prompts import PromptTemplate


# Warning filter
warnings.filterwarnings('ignore', "TypedStorage is deprecated", UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def init():
    # Init docstore, retriever, write docs, gpt, return a pipeline for doc search
    document_store, loaded = init_store()
    retriever = init_retriever(document_store)
    if not loaded:
        write_docs(document_store, retriever)
    init_gpt()
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

def write_docs(document_store, retriever):
    # Get dataframe with columns "question", "answer" and some custom metadata
    df = pd.read_csv("qna1.csv")
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

def get_response(pipe, query):
    prediction = query_faiss(query, pipe) 
    # Generate prompt from related docs
    prompt, scores, alts, CIDs, source_links, source_filenames, SMEs, best_sme = create_prompt(query, prediction)
    output, conf = call_gpt(prompt, scores, alts)
    return output, conf, prompt, CIDs, source_links, source_filenames, SMEs, best_sme

def query_faiss(query, pipe):
    return pipe.run(query=query, params={"Retriever": {"top_k": 5}})

# Create prompt template
def create_prompt(query, prediction):  ### may add a parameter "Short", "Yes/No", "Elaborate", etc. for answer preferences
    print("Creating prompt")
    prompt = PromptTemplate(input_variables=["prefix", "question", "context"],
                            template="{prefix}\nQuestion: {question}\n Context: {context}\n")

    # Provide instructions/prefix
    prefix = """You are an assistant for the Information Security department of an enterprise designed to answer security questions professionally. Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, answer'. Use the answers within the context to answer the original question in a concise manner with explanation. Just at the end, list the question IDs of the answers you referenced to formulate your response."""
    # Create context
    context = ""
    scores = {}
    SMEs, sme_dict, source_filenames, source_links, CIDs = [], {}, [], [], []  # SMEs are dictionary in this function to allow us to more easily get best_sme later
    alts = []
    count = 0
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
        if count < 3:
            alts.append(answer.meta["cid"])
            count+=1
    # print("Scores:\n", scores)
    # Get the question ID with the highest score, use as primary SME
    max_id = max(scores, key=scores.get)
    best_sme = sme_dict[max_id]
    return prompt.format(prefix=prefix, question=query, context=context), scores, alts, CIDs, source_links, source_filenames, SMEs, best_sme 
    

# Call openai API
def call_gpt(prompt,scores, alts):  # returns None as confidence if no sources used for prompt
    deployment_name = 'immerse-3-5'
    response = openai.Completion.create(
        engine=deployment_name,
        prompt=(f"Original Question: {prompt}\n"
                "Answer:"
                ),
        max_tokens=1000,
        n=1,
        temperature=0.3,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    output = response.choices[0].text.split('\n')[0]
    ids = re.findall("CID\d+", output)
    ids = list(set(ids))
    output = re.sub("\(?(CID\d+),?\)?|<\|im_end\|>|\[|\]", "", output)

    # Handle case where gpt doesn't output sources in prompt
    if ids is None or len(ids) == 0:
        alternates = ""
        for i in alts:
            alternates += f"{i.strip()}\n"
        return f"{output}\nBelow are some possible sources for reference", "**Confidence:** N/A"

    output = output[ 0 : output.rindex(".") + 1]
    confidence = utils.compute_average(ids,scores)
    conf_str = f"\n\n**Confidence Score:** {confidence:.2f}%"
    return output, conf_str

def init_gpt():
    with open('gpt-config.json') as user_file:
        content = json.load(user_file)
    
    if content is None:
        raise Exception("Error reading config")

    openai.api_key = content["api_key"]
    openai.api_type = content["api_type"] 
    openai.api_version = content["api_version"]
    openai.api_base = content["api_base"]

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
    # Initialize pipline
    pipe = init()
    file_uploaded=False

    # Init UI Slots
    st.header("Ask a Question:")
    # file_upload = st.checkbox("Upload questions from file")
    # if file_upload:
    #     questions_file = st.file_uploader("Upload a CSV or Excel file (each cell a question, max 50 questions)", type=['csv', 'xlsx', 'txt'])
    #     if questions_file is not None:
    #         questions, errCode = utils.read_questions(questions_file)
    #         if errCode==1:
    #             st.error("Emtpy file")
    #         elif errCode ==2:
    #             st.error("File type not supported. Please upload a CSV or Excel file.")
    #         else:
    #             file_uploaded=True
    #             questions = questions[:50]
    #         print(questions)
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
        try:
            # Query database, generate prompt from related docs, initialize gpt-3
            output, conf, prompt, CIDs, source_links, source_filenames, SMEs, best_sme = get_response(pipe, query) 
            CIDs, source_links, source_filenames, SMEs = utils.remove_duplicates(CIDs, source_links, source_filenames, SMEs)
            # Feed prompt into gpt, store query & output in session state
            st.session_state.responses.append(query)
            st.session_state.responses.append(output)
            
            response_header_slot.markdown(f"**Answer:**\n")
            response_slot.write(output)  
            with response_copy.expander('Copy response'):
                st.write("Copied response!")
                pc.copy(output) 

            # Display confidence, sources, SMEs
            confidence_slot.markdown(conf)
            sources_header.markdown(f"**Sources:**")
            # create a markdown table
            markdown_table = "| CID | SME | File Name |\n| --- | --- | --- |\n|"
            for i in range(len(CIDs)):
                markdown_table += "[{0}]({1}) | {2} | {3} |\n|".format(CIDs[i], source_links[i], SMEs[i], source_filenames[i])
            sources_slot.write(markdown_table, unsafe_allow_html=True)
            best_sme = utils.parse_sme_name(best_sme)
            best_sme_slot.markdown(f"**SME:** {best_sme} ")
            # Draft email option
            with draft_email.expander('Draft an email to the SME'):
                if draft_email.expander:
                    print("Drafting email...")
                    emailHeader.markdown("### Email To SME:")
                    prompt = f"Please write a brief and professional business email to someone named {best_sme} asking {query}. Include only the text of the email in your response, not any sort of email address, and should be formatted nicely. The email should start with Subject: __ \n\nand end with the exact string \n\n'[Your Name]'."
                    response = openai.Completion.create(
                        engine='immerse-3-5',
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=400,
                        frequency_penalty=0.0,
                        presence_penalty=0,
                    )
                    email_response = response.choices[0].text
                    print(email_response)
                    subject_index = email_response.find("Subject:")
                    name_index = email_response.find("[Your Name]")
                    email_response = email_response[subject_index:name_index+len("[Your Name]")].strip()
                    emailContent.write(email_response)
        except:
            print("Error initializing var")
            traceback.print_exc()
        # print(st.session_state.responses)

if __name__ == "__main__": 
    main()



