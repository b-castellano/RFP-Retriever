# RFP Retriever

## A concise and accurate Q&A bot for RFPs (requests for proposals), sourcing data from RFPIO and sharepoint

### Steps to run:

1. After cloning, enter:
    pip3 -r requirements.txt
into the terminal

2. Add a file called gpt-config.json, with the following contents: /
    { /
        "api_key": "<your_api_key_here>", /
        "api_type": "azure", /
        "api_version": "2023-06-01-preview", /
        "api_base": "https://immerse.openai.azure.com/" /
    }

3. In the terminal, enter:
    streamlit run app.py

