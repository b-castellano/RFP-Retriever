## NOTE: Bot responses can be changed by adding documents with valuable QA pairs to RFP. Consider this option if you are attempting to tweak responses

# General
import warnings
import os
import pandas as pd
import traceback
import threading
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
import concurrent.futures
from custom_html import custom_response

# External Files
import utils
import ps

# Streamlit
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Warning filter
warnings.filterwarnings('ignore', "TypedStorage is deprecated", UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

### Setup session storage
st.session_state.responses = []
st.session_state.data = []
st.session_state.submit = False
st.session_state.single_question = True

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
    rfp_pipe, sp_pipe = ps.init()

    ### Setup session storage
    if "responses" not in st.session_state:
        st.session_state.responses = []

    # Init UI Header/File Upload
    st.header("Ask a Question:")
    file_upload = st.checkbox("Upload questions from file")
    if st.session_state.single_question == False:
        file_upload = st.checkbox("Upload questions from file", value=True)

    # Read questions from file uploaded and gather row data
    questions = []
    rows = [] ## Row indexes
    if file_upload:
        st.session_state.single_question = False
        query = st.empty()
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
    if not file_upload:
        query = st.text_input("RFP/Security-Related")
    submitted = st.button("Submit")

    response_header_slot = st.empty()
    container = st.container()

    confidence_slot = st.empty()
    sources_header = st.empty()
    sources_slot = st.empty()
    best_sme_slot = st.empty()

    # draft_email = st.empty()
    # email_header = st.empty()
    # email_content = st.empty()

    if query and isinstance(query, str) or submitted: ## If user submits a question
        try:
            if isinstance(query, str) and query.strip() != "":  ## Check for empty user query
                questions.append(query)
                print("Appending")

            if len(questions) == 1: ## Single question case
                # Get response from rfp-retriever and assign 
                response = ps.get_response(rfp_pipe,sp_pipe, questions[0],history=st.session_state.responses)
                output = response.answer
                cids = response.cids
                smes = response.smes
                source_links = response.source_links
                best_sme = response.best_sme

                # Add query and output to front end
                # st.session_state.responses.append(questions[0])
                # st.session_state.responses.append(output)
                st.session_state.responses.append({"question":questions[0],"answer":output})

                response_header_slot.markdown(f"**Answer:**")
                with container:
                    custom_response(output)

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

                # Write drafted email
                # draft_email.expander('Draft an email to the SME')
                # if draft_email.expander:
                #     email_text = utils.get_email_text(query, best_sme)
                #     email_header.markdown("### Email to SME:")
                #     email_content.write(email_text)
                questions.clear()

            elif len(questions) > 1: # Multiple questions case
                print(f"\n\nQuestion length is: {len(questions)}\n\n")
                st.session_state.submit = True

                # Initialize empty lists for answers, CIDs, source_links, SMEs, and confidences
                answers, cids, source_links, best_smes, confidences = [], [], [], [], []
                # Initiate variables for multi-threading
                lock = threading.Lock()
                threads = []
                stop_event = threading.Event()
                stop = False

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
                progress_text = f"Questions being answered. Please wait. ({num_complete[0]} / {len(questions)} complete)"
                progress_bar = st.progress((num_complete[0] / len(questions)), text=progress_text)

                # Thread creation
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    # Schedule a thread for each question in the sheet
                    for i, question in enumerate(questions):
                        threads.append(executor.submit(ps.get_responses, rfp_pipe, sp_pipe,questions,
                                                    answers, cids,
                                                    source_links,
                                                    best_smes,
                                                    confidences,
                                                    i,
                                                    lock,
                                                    num_complete,
                                                    progress_text,
                                                    progress_bar))

                        # Enable multi-threading for Streamlit (used for progress bar)
                        for thread in executor._threads:
                            add_script_run_ctx(thread)

                        # Check if the "Stop" button is clicked by the user or if the expander is expanded
                        # if stop_button_clicked:
                        #     with lock:
                        #         print("Stopping early")
                        #         print(answers)
                        #         stop_event.set()
                        #         progress_text = "Halted Execution. Current progress still downloadable"
                        #     break


                # Create dataframe for display
                df = pd.DataFrame({"Question": questions, "Answer": answers, "Confidence": confidences, "SMEs": best_smes, "Source Links": source_links})
                st.session_state.data.append(df)

                # Second dataframe for hyper links on excel <-- Weird issues with html and hyper dataframes merging together
                df_2 = pd.DataFrame({"Question": questions, "Answer": answers, "Confidence": confidences, "SMEs": best_smes, "Source Links": source_links})
                sources_slot.write(df)
                
                # # Copy button for only question, answer columns
                # copy_qa_button = Button(label="Copy Questions, Answers only")
                # df_copy = df.iloc[:, :2].to_csv(sep='\t')  # Select first two columns and convert to CSV
                # copy_qa_button.js_on_event("button_click", CustomJS(args=dict(df=df_copy), code=""" navigator.clipboard.writeText(df); """))
                # copy_qa_button.css_classes = ["streamlit-button"]

                # # Copy button for all of df
                # copy_all_button = Button(label="Copy All")
                # copy_all_button.js_on_event("button_click", CustomJS(args=dict(df=df.to_csv(sep='\t')), code="""
                #     navigator.clipboard.writeText(df);
                #     """))
                # copy_all_button.css_classes = ["streamlit-button"]

                # st.bokeh_chart(copy_qa_button)
                # st.bokeh_chart(copy_all_button)
                
                # Create file and html table from dataframe and append to session state
                file = df.to_csv()
                hyper_df = utils.to_hyperlink(df_2, cids)
                df_html = utils.to_html(df, cids)
                hyper_file = utils.to_excel(hyper_df, rows)
                st.session_state.data.append(file)
                st.session_state.data.append(df_html)
                st.session_state.data.append(hyper_file)

                # Display completed over progress bar when done 
                if (num_complete[0] / len(questions)) == 1 and "" not in answers:
                   progress_text = "Completed."
                   progress_bar.progress((num_complete[0] / len(questions)), text=progress_text)
            else:
                st.error("No questions detected")

        except:
            print("Error initializing var")
            traceback.print_exc()

if __name__ == "__main__": 
    main()