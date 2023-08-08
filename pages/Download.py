import streamlit as st
import pandas as pd
import utils

st.set_page_config(page_title="Document Download", page_icon="📁")

# Sidebar options
with st.sidebar:
    st.title("Document Downloader")
    st.image("download.jpg")

st.markdown("## Download ##")
try: ## Check for errors
    if st.session_state.submit == True: ## If multiple questioin file is submitted

        # Assign data variables from session state
        df = st.session_state.data[0]
        file = st.session_state.data[1]
        df_html = st.session_state.data[2]
        hyper_file = st.session_state.data[3]

        # Download buttons UI
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(label='Download Excel', data=hyper_file, file_name="text_2.xlsx")
        with col2:
            st.download_button("Download CSV", data=file, file_name="test.csv", mime="txt/csv")
        
        # Get unanswerd questions and SMEs
        SMEs = utils.get_SMEs(df)
        options = ['None']

        # If unanswered questions
        if len(SMEs) != 0:
            for SME in SMEs.keys():
                options.append(SME)

        # SME option selectbox
        SME = st.selectbox(
            'Select a SME',
            (options),
        )

        # Diplay dropdown of SMEs and relevant unanswered questions
        if SME != 'None':
            with st.expander(f'Unanswered Questions: {len(SMEs[SME])}'):
                st.write(SMEs[SME])

        # Display HTML table with hyper links
        st.components.v1.html(df_html.to_html(escape = False), width = 880, height = 500, scrolling = True)

    # If no file is submitted
    elif st.session_state.submit == False:
            st.markdown("Submit a file to download the responses")

except: ## If errors
    st.markdown("Submit a file to download the responses")