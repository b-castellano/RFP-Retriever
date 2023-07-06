# VectorDB & Retriever Front-End (Demo)

## RFP QA Bot




### Should have your GPT API key in a separate file called init_gpt.py similar to the following:
``` py
import openai
def init_gpt():
    openai.api_key = "<your key here>"
    openai.api_type = "azure"
    openai.api_version = "2023-06-01-preview"
    openai.api_base = "https://immerse.openai.azure.com/"
```
