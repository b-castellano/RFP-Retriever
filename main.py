import pandas as pd
from datasets import load_dataset
import torch
import openai
from tqdm.auto import tqdm  # progress bar

from haystack.nodes import EmbeddingRetriever
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers


import langchain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

loaded = False
try:
    document_store = FAISSDocumentStore.load(index_path="my_faiss_index.faiss")
    loaded = True
except:
    document_store = FAISSDocumentStore(
        similarity="cosine",
        embedding_dim=768,
        duplicate_documents = 'overwrite'
    )

print(document_store.metric_type)              # should output "cosine"
print(document_store.get_document_count())     # should output "0"
print(document_store.get_embedding_count())    # should output "0"

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

# Example Prompt Answers
examples = [
    {
    "question": "Does your company have an access control policy?",
    "answer":"Yes that is correct. /n Confidence Interval: %95.3,
    Sources:
    * Source 1
    * Source 2
    * Source 3 "
    }
]

example_prompt = PromptTemplate(input_variables=["question", "answer", "ci", "sources"], template="""Question: {question}\n Answer: {answer}\n Confidence Interval: {ci}\n Sources: {sources}""")

prefix = """The following are examples of questions and answers related to the security of a company. The responses are professional. Here are a few examples:"""
suffix="""
 If the question cannot be answered with the information reply with 'Question cannot be answered.'
    Question: {question}\n
Please use information from the following context documents in the response and list the question IDs as sources in bullet points.
    Context: {context}
Also include the confidence interval at the end of the answer.
    Confidence Interval: {ci}
    Answer: """

# Prompt Template Generation
fs_prompt = FewShotPromptTemplate (
    examples=examples,
    example_prompt=example_prompt,
    suffix=suffix,
    input_variables=["question", "context", "ci"]
)

openai.api_key = "dd9d2682f30f4f66b5a2d3f32fb6c917"
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = "https://immerse.openai.azure.com/"
deployment_name='immerse-3-5'

while True:

    query = input("What question would you like to ask? (Type \"STOP\" to exit): ")
    if query == "STOP":
        break
    
    prediction = pipe.run(query=query, params={"Retriever": {"top_k": 4}})

    # print_answers(prediction, details="medium")
    # print(prediction["answer"][0].meta)

    prompt_question = query
    # print(f"Prompt Question: {prompt_question}")

    # Prompt context and score count
    total_score = 0
    count = 0
    prompt_context = ""
    for answer in prediction["answers"]:
        # print(answer.meta["Question ID"])

        # Score calculations
        total_score += answer.score
        count += 1

        prompt_context += answer.meta["answer"]
    total_score /= count
    # print(f"Prompt Context:\n {prompt_context}")
    print(f"Mean Score: {total_score}")

    print("Generating prompt...")
    print("PROMPT:\n=======================",fs_prompt.format(question=prompt_question, context=prompt_context, ci=total_score), "\n=======================")
   
    question = fs_prompt.format(question=prompt_question, context=prompt_context, ci=total_score)
    response = openai.Completion.create(
        engine=deployment_name,
        prompt=(f"Question: {question}\n"
                "Answer:"
                ),
        max_tokens=100,
        n=1,
        top_p=0.7,
        temperature=0.5,
        frequency_penalty= 0.5,
        presence_penalty= 0.2
    )
    gptResponse = response.choices[0].text.split('\n')[0]
    print("OUTPUT:\n=======================",gptResponse,"\n=======================")










