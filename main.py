import warnings
from tqdm.auto import tqdm  # progress bar
from datasets import load_dataset
import pandas as pd
import torch
import openai
import os
import random

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack import Document
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers

import langchain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

# Warning filter
warnings.filterwarnings('ignore', "TypedStorage is deprecated", UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

loaded = False
try:
    document_store = FAISSDocumentStore.load(index_path="my_faiss_index.faiss")
    loaded = True
except:
    document_store = FAISSDocumentStore(
        similarity="cosine",
        embedding_dim=768,
        duplicate_documents='overwrite'
    )

'''
print(document_store.metric_type)              # should output "cosine"
print(document_store.get_document_count())     # should output "0"
print(document_store.get_embedding_count())    # should output "0"
'''

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    model_format="sentence_transformers"
)

if not loaded:
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

print("docs:", document_store.get_document_count())
print("embeddings:", document_store.get_embedding_count())

pipe = FAQPipeline(retriever=retriever)

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
template_dylan = """"You are an assistant for the Information Security department of an enterprise designed to answer security questions in a professional manner. 
Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, confidence score, and answer'. 
Use the answers within the context to formulate your response in under two hundred words. 
In addition, list the referenced question ID in parenthesis after the portion of your response using the associated answer.
Include the average confidence score of the referenced context portions at the end of your answer."

Question: {question}

Context: {context}
"""

gpt_template_dylan = PromptTemplate (
    input_variables=["question", "context"],
    template=template_dylan
)

# Modified FewShotTemplate --> Issues with too much text.
fs_template_modified = FewShotPromptTemplate (
    examples=examples,
    example_prompt=example_prompt,
    suffix="""
    You are an assistant for the Information Security department of an enterprise designed to answer security questions in a professional manner.
    Use the examples above as a reference for formating.
    Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, confidence score, and answer'.
    Use the answers within the context to formulate your response in under two hundred words. 
    In addition, list the referenced question ID in parenthesis after the portion of your response using the associated answer.
    Include the average confidence score of the referenced context portions at the end of your answer."

    Question: {question}

    Context: {context}
    """,
    input_variables=["question", "context"]
)

# Open AI Information
openai.api_key = "dd9d2682f30f4f66b5a2d3f32fb6c917"
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = "https://immerse.openai.azure.com/"
deployment_name='immerse-3-5'

# Manual Question Storage
questions = [
"Does your company have an access control policy?",
"Is company information backed up regularly?",
"Do your security policies require standard encryption solutions and protocols?",
"Does your company monitor for unauthorized actions?"
]
n = 0

while True:

    #query = input("What question would you like to ask? (Type \"STOP\" to exit): ")
    #if query == "STOP":
    #    break

    # Get n questions for mass testing
    '''
    df = pd.read_csv("qna.csv")
    max = len(list(df["question"].values))
    i = random.randint(0, max+1)
    query = df["question"].values[i]
    '''

    df = pd.read_csv("qna.csv")
    query = df["question"].values[n]
    prediction = pipe.run(query=query, params={"Retriever": {"top_k": 4}})
    prompt_question = query

    # Prompt context and score count
    total_score = 0
    count = 0
    prompt_context = ""
    prompt_ids = ""
    for answer in prediction["answers"]:
        prompt_context += "Question ID: {ID}\n Content: {content}\n Confidence Score: {ci}\n".format(ID=answer.meta["question ID"], content=answer.meta["answer"], ci=answer.score)
        prompt_ids += "{ID}\n".format(ID=answer.meta["question ID"])

        total_score += answer.score
        count += 1

    # Check for no reliable answer
    if count == 0:
        total_score = 0
    else:
        total_score /= count
    print(f"{total_score}")

    print("Generating prompt...")
    # Prompt Storage
    fs_prompt = fs_template.format(question=prompt_question, context=prompt_context, ci=total_score)
    gpt_prompt = gpt_template.format(context=prompt_context, question=prompt_question,ci=total_score,ID=prompt_ids)
    gpt_prompt_simple = gpt_template_simple.format(question=prompt_question, context=prompt_context)
    gpt_prompt_dylan = gpt_template_dylan.format(question=prompt_question, context=prompt_context)
    fs_prompt_modified = fs_template_modified.format(question=prompt_question, context=prompt_context)

    full_prompt = gpt_prompt_dylan

    # AI Response Prompt
    print("PROMPT:\n=======================\n",full_prompt,"\n=======================\n")
    response = openai.Completion.create(
        engine=deployment_name,
        prompt=(full_prompt),
        max_tokens=1000,
        n=1,
        top_p=0.7,
        temperature=0.3,
        frequency_penalty= 0.5,
        presence_penalty= 0.2
    )
    gptResponse = response.choices[0].text.split('\n')[0]
    print("OUTPUT:\n=======================\n",gptResponse,"\n=======================\n")

    # Comparision Output
    output = """
    {answer}
    Confidence Score: {ci}
    Sources:
{IDs}
    """
    
    # Check for no reliable answer
    if (total_score == 0):
        output = "No reliable answer."
    else:
        output = output.format(answer=gptResponse, ci=total_score, IDs=prompt_ids)
    print(output)
    
    # Amount of questions to run through
    n += 1
    if (n == 3):
        break