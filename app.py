## NOTE: Bot responses can be changed by adding documents with valuable QA pairs to RFP. Consider this option if you are attempting to tweak responses

# General
import warnings
import os
import numpy as np
import pandas as pd
import openai
import traceback
import pyperclip as pc
import threading
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
import fontawesome as fa


# External Files
import utils
import ps
from response import Response

# Streamlit
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit.runtime.scriptrunner import add_script_run_ctx

import concurrent.futures

# Warning filter
warnings.filterwarnings('ignore', "TypedStorage is deprecated", UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    pipe = ps.init()

    # Init UI Header/File Upload
    st.header("Ask a Question:")
    file_upload = st.checkbox("Upload questions from file")
    file_prev = st.empty()

    # Read questions from file uploaded and gather row data
    questions = []
    rows = [] ## Row indexes
    if file_upload:
        questions_file = st.file_uploader("Upload a CSV or Excel file (each cell a question, max 50 questions)", type=['csv', 'xlsx'])
        if questions_file is not None:
            questions, errCode, rows = utils.read_questions(questions_file)
            if errCode==1:
                st.error("Empty file")
            elif errCode ==2:
                st.error("File type not supported. Please upload a CSV or Excel file.")
            else:
                questions = questions[:300] ## Max amount of questions allowed 

    # UI Elements
    query = st.text_input("RFP/Security-Related")
    submitted = st.button("Submit")

    response_header_slot = st.empty()
    response_slot = st.empty()
    response_copy = st.empty()

    confidence_slot = st.empty()
    sources_header = st.empty()
    sources_slot = st.empty()
    sources_slot_copy_button = st.empty()
    best_sme_slot = st.empty()

    draft_email = st.empty()
    email_header = st.empty()
    email_content = st.empty()

    if query or submitted: ## If user submits a question
        try:
            if query.strip() != "":  ## Check for empty user query
                questions.append(query)

            if len(questions) == 1: ## Single question case

                # Get response from rfp-retriever and assign 
                response = ps.get_response(pipe,questions[0])
                output = response.answer
                cids = response.cids
                smes = response.smes
                source_links = response.source_links
                best_sme = response.best_sme

                # Add query and output to front end
                st.session_state.responses.append(questions[0])
                st.session_state.responses.append(output)

                # Write response
                # response_slot.write(f
                # '''
                # **Answer:**

                #     {output}
                # ''')  
                response_header_slot.markdown(f"**Answer:**")
                response_slot.write(f"<code>\n{output}\n</code>", unsafe_allow_html=True)

                # Display confidence, sources, SMEs
                confidence_slot.markdown(f"**Confidence Score:** {response.conf}")
                sources_header.markdown(f"**Sources:**")

                # Create a markdown table
                markdown_table = "| CID | SME |\n| --- | --- |\n|" 
                
                for i in range(len(cids)):
                    markdown_table += "[{0}]({1}) | {2}|\n".format(cids[i], source_links[i], smes[i]) 
                sources_slot.write(markdown_table, unsafe_allow_html=True)

                # Write most relevant SME
                best_sme_slot.markdown(f"**SME:** {best_sme} ")

                # # Write drafted email
                # with draft_email.expander('Draft an email to the SME'):
                #     if draft_email.expander:
                #         email_text = utils.get_email_text(query, best_sme)
                #         email_header.markdown("### Email to SME:")
                #         email_content.write(email_text)
                questions.clear()

            elif len(questions) > 1: # Multiple questions case
                print(f"\n\nQuestion length is: {len(questions)}\n\n")

                # Initialize empty lists for answers, CIDs, source_links, SMEs, and confidences
                answers, cids, source_links, best_smes, confidences = [], [], [], [], []

                # Initiate variabels for multi-threading
                lock = threading.Lock()
                stop_flag = False
                threads = []

                for i, question in enumerate(questions):
                    with lock:

                        # Append empty strings and lists to answers, cids, source_links, source_filenames, and SMEs
                        answers.append("")
                        cids.append([])
                        source_links.append([])
                        best_smes.append([])
                        confidences.append(0)
                    
                # Progress bar
                num_complete = [0]
                progress_text = "Questions being answered, please wait."
                progress_bar = st.progress((num_complete[0] / len(questions)), text=progress_text)

                # Thread creation
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    for i, question in enumerate(questions):
                        threads.append(executor.submit(ps.get_responses, pipe, questions, answers, cids, source_links,  best_smes, confidences, i, lock, num_complete, progress_text, progress_bar))
                        for thread in executor._threads:
                            add_script_run_ctx(thread)

                #  Download file for multiple questions answers
                st.markdown("### Download")

                # print(f"questions: {len(questions)}")
                # print(f"answers: {len(answers)}")
                # print(f"confidences: {len(confidences)}")
                # print(f"best_SMEs: {len(best_SMEs)}")
                # print(f"source_links: {len(source_links)}")
                # print(f"source_filenames: {len(source_filenames)}")
                # print(f"questions: {questions}")
                # print(f"answers: {answers}")
                # print(f"confidences: {confidences}")
                # print(f"best_SMEs: {best_SMEs}")
                # print(f"source_links: {source_links}")
                # print(f"source_filenames: {source_filenames}")

                # Format for excel
                a = {'Question' : questions ,'Answer' : answers , 'Confidence': confidences , 'SMEs': best_smes, 'Source Links': source_links}
                df = pd.DataFrame.from_dict(a, orient='index')
                df = df.transpose()
                #df = pd.DataFrame({"Question": questions, "Answer": answers, "Confidence": confidences, "SMEs"sme, "Source Links": source_links, "Souce Filenames": source_filenames})
                sources_slot.write(df)

                # Copy button for only question, answer columns
                copy_qa_button = Button(label="Copy Questions, Answers only")
                df_copy = df.iloc[:, :2].to_csv(sep='\t') # Select first two columns and convert to CSV
                copy_qa_button.js_on_event("button_click", CustomJS(args=dict(df=df_copy), code=""" navigator.clipboard.writeText(df); """))
                copy_qa_button.css_classes = ["streamlit-button"]

                # Copy button for all of df
                copy_all_button = Button(label="Copy All")
                copy_all_button.js_on_event("button_click", CustomJS(args=dict(df=df.to_csv(sep='\t')), code="""
                    navigator.clipboard.writeText(df);
                    """))
                copy_all_button.css_classes = ["streamlit-button"]

                # Download buttons for csv/excel, put buttons on UI
                file = utils.to_excel(df, rows)
                col1, col2 = st.columns(2)
                with col1:
                    st.bokeh_chart(copy_qa_button)
                    st.download_button(label='Download Excel', data=file, file_name="text_2.xlsx")
                with col2:
                    st.bokeh_chart(copy_all_button)
                    st.download_button("Download CSV", data=df.to_csv(), file_name="test.csv", mime="txt/csv")

            else:
                st.error("No questions detected")

        except:
            print("Error initializing var")
            traceback.print_exc()

if __name__ == "__main__": 
    main()