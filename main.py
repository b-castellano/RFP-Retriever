# General
import warnings
from tqdm.auto import tqdm  # progress bar
from datasets import load_dataset
import pandas as pd
import torch
import openai
import os
import random
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

# Warning filter
warnings.filterwarnings('ignore', "TypedStorage is deprecated", UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## TEMPLATE STORAGE
# FewShotPrompt Template (WORST) --> To much information?
examples = [
    {
    "question": "Does your company have an access control policy?",
    "answer":
"""
   Yes, that is correct.     
""",
    "ci": "95.3%",
    "sources":
"""
* Source 1
* Source 2
* Source 3
"""
    },
    {
    "question": "Test 2 Question",
    "answer":
"""
    Test 2 answer.   
""",
    "ci": "50.3%",
    "sources":
"""
* Source 1
* Source 2
"""
    }
]
example_prompt = PromptTemplate(input_variables=["question", "answer", "ci", "sources"], template="Question: {question}\n Answer: {answer}\n Confidence Interval: {ci}\n Sources: {sources}")

fs_template = FewShotPromptTemplate (
    examples=examples,
    example_prompt=example_prompt,
    suffix="""
Generate a coeherent response based off the following question using the above examples as formating reference. If the question cannot be answered with the information reply with 'Question cannot be answered.'
    Question: {question}\n
Please use information from the following context documents in the response and list the question IDs as sources in bullet points.
    Context: {context}
Also include the confience interval at the end of the answer.
    Confidence Interval: {ci}
    Answer:""",
    input_variables=["question", "context", "ci"]
)

# Normal Template --> Too simple of answer, sometimes list confidence intervals, lists sources.
template = """Give a coherent response to the question based on the context below then include the confidence score and question IDs. Fill in the required information in the empty spaces below.

Question: {question}
Context: {context}
Confidence Score: {ci}
Question IDs: {ID}

Provde an answer then put confidence score after and put all question IDs as sources from the above information.

Output:
"""

gpt_template = PromptTemplate (
    input_variables=["context","question","ci","ID"],
    template=template
)

# Simple Template --> Simple answer
template_simple = """Give a coherent and thorough response to the question based on the context below.

Question: {question}

Context: {context}

Answer:
"""

gpt_template_simple = PromptTemplate (
    input_variables=["question","context"],
    template=template_simple
)

# Dylan prefix tempalte (BEST) --> Outputs solid answer, lists sources, gives relatively accurate confidence interval.
template_dylan = """You are an assistant for the Information Security department of an enterprise designed to answer security questions in a professional manner.
Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID and answer'.
Use the answers within the context to formulate a concise response.
List the question IDs of the answers you referenced at the end of your response in this form: [...,...]"
Question: {question}

Context: {context}
"""

gpt_template_dylan = PromptTemplate (
    input_variables=["question", "context"],
    template=template_dylan
)

# JSON Template ---> Does not work at all
template_json = """"You are an assistant for the Information Security department of an enterprise designed to answer security questions in a professional manner. 
Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, confidence score, and answer'. 
Use the answers within the context to formulate a concise response.
You must add at the end of the response the referenced question IDs in the form of a list."

Question: {question}

Context: {context}

Answer in a JSON like this:
"Answer": ...,
"Sources": ...
"""

gpt_template_json = PromptTemplate (
    input_variables=["question", "context"],
    template=template_json
)


# Output Parser Template -> Only lists sources
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

list_template = """"You are an assistant for the Information Security department of an enterprise designed to answer security questions in a professional manner. 
Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, confidence score, and answer'. 
Use the answers within the context to formulate a concise response. 
At the end of the response give the referenced question IDs in the form of a list."
Question: {question}

Context: {context}

Make a list of the referenced question IDs
List:\n {format_instructions}
"""

gpt_list_template = PromptTemplate (
    input_variables=["question", "context"],
    template=list_template,
    partial_variables={"format_instructions": format_instructions}
)

# Modified FewShotTemplate --> Issues with too much text.
fs_template_modified = FewShotPromptTemplate (
    examples=examples,
    example_prompt=example_prompt,
    suffix="""
    You are an assistant for the Information Security department of an enterprise designed to answer security questions in a professional manner.
    Use the examples above as a reference for formating.
    Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, confidence score, and answer'.
    Use the answers within the context to formulate a concise response.
    At the end of the response give the referenced question IDs in the form of a list as well as the average confidence score for the referenced context portions.
    "
    Question: {question}

    Context: {context}
    """,
    input_variables=["question", "context"]
)

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
    return pipe.run(query=query, params={"Retriever": {"top_k": 4}})

def create_prompt(query, prediction):
    prompt = gpt_template_dylan
    IDs = {}

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
    main()