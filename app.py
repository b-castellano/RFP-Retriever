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

# External Files
import utils
import ps

# Streamlit
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

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

    # Init UI Slots
    st.header("Ask a Question:")

    file_upload = st.checkbox("Upload questions from file")
    file_prev = st.empty()

    # Read questions from file uploaded and gather row data
    questions = []
    rows = []
    if file_upload:
        questions_file = st.file_uploader("Upload a CSV or Excel file (each cell a question, max 50 questions)", type=['csv', 'xlsx'])
        if questions_file is not None:
            questions, errCode, rows = utils.read_questions(questions_file)
            if errCode==1:
                st.error("Emtpy file")
            elif errCode ==2:
                st.error("File type not supported. Please upload a CSV or Excel file.")
            else:
                questions = questions[:50]  

    # Prompt options --> May remove
    # options = np.array(["Short", "Regular", "Elaborate", "Yes/No"])
    # selected_option = st.selectbox('Desired answer type:', options=options, index=0)
    # nda_status = st.checkbox("NDA Signed?")

    # Fill in text boxes on page
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

                # Get response from rfp-retriever
                output, conf, CIDs, source_links, source_filenames, SMEs, best_sme = ps.get_response(pipe, questions[0]) 
                print(len(CIDs))
                # Add query and output to front end
                st.session_state.responses.append(questions[0])
                st.session_state.responses.append(output)

                # Write response
                response_header_slot.markdown(f"**Answer:**\n")
                response_slot.write(output)  

                # Copy response
                with response_copy.expander('Copy response'):
                    st.write("Copied response!")
                    pc.copy(output) 

                # Display confidence, sources, SMEs
                confidence_slot.markdown(f"**Confidence Score:** {conf}")
                sources_header.markdown(f"**Sources:**")

                # Create a markdown table
                markdown_table = "| CID | SME | File Name |\n| --- | --- | --- |\n|" 
                for i in range(len(CIDs)):
                    markdown_table += "[{0}]({1}) | {2} | {3} |\n|".format(CIDs[i], source_links[i], SMEs[i], source_filenames[i]) 
                sources_slot.write(markdown_table, unsafe_allow_html=True)

                # Write most relavent SME
                best_sme = utils.parse_sme_name(best_sme)
                best_sme_slot.markdown(f"**SME:** {best_sme} ")

                # Write drafted email
                with draft_email.expander('Draft an email to the SME'):
                    if draft_email.expander:
                        email_text = utils.get_email_text(query, best_sme)
                        email_header.markdown("### Email to SME:")
                        email_content.write(email_text)
                questions.clear()

            elif len(questions) > 1: # Multiple questions case
                print(f"\n\nQuestion length is: {len(questions)}\n\n")

                # Initialize empty lists for answers, CIDs, source_links, source_filenames, SMEs, and confidences
                answers, CIDs, source_links, source_filenames, best_SMEs, confidences = [], [], [], [], [], []

                # Initiate variabels for multi-threading
                lock = threading.Lock()
                stop_flag = False
                threads = []

                for i, question in enumerate(questions):
                    with lock:

                        # Append empty strings and lists to answers, CIDs, source_links, source_filenames, and SMEs
                        answers.append("")
                        CIDs.append([])
                        source_links.append([])
                        source_filenames.append([])
                        best_SMEs.append([])
                        confidences.append(0)

                    # Start a new thread for each question
                    thread = threading.Thread(target=ps.get_responses, args=(pipe, questions, answers, CIDs, source_links, source_filenames, best_SMEs, confidences, i))
                    thread.start()
                    threads.append(thread)
                    thread.join()

                # Wait for threads, timeout threads if they take too long
                for thread in threads:
                    thread.join(timeout=18)

                    if thread.is_alive():

                        # Set stop flag to signal thread to stop gracefully
                        stop_flag = True
                        thread.join()

                        # Start a new thread to replace the stopped thread
                        with lock:
                            answers[i] = ""
                            CIDs[i] = []
                            source_links[i] = []
                            source_filenames[i] = []
                            best_SMEs[i] = []
                            confidences[i] = 0
                        new_thread = threading.Thread(target=ps.get_responses, args=(pipe, questions, answers, CIDs, source_links, source_filenames, best_SMEs, confidences, i, lock, stop_flag))
                        new_thread.start()
                        threads.append(new_thread)

                #  Download file for multiple questions answers
                st.markdown("### Download")

                source_links = list(filter(None,source_links))
                source_filenames = list(filter(None,source_filenames))
                print(source_links)
                print(source_filenames)

                # Format for excel
                df = pd.DataFrame({"Question": questions, "Answer": answers, "Confidence": confidences, "SMEs": best_SMEs, "Source Links": source_links, "Souce Filenames": source_filenames})
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

                # st.download_button(label='Download Excel', data=file, file_name="text_2.xlsx")
                # st.download_button("Download CSV", data=df.to_csv(), file_name="test.csv", mime="txt/csv")

            else:
                st.error("No questions detected")

        except:
            print("Error initializing var")
            traceback.print_exc()

if __name__ == "__main__": 
    main()