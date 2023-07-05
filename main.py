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
        duplicate_documents='overwrite'
    )


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


openai.api_key = "dd9d2682f30f4f66b5a2d3f32fb6c917"
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = "https://immerse.openai.azure.com/"
deployment_name = 'immerse-3-5'

# while True:
query = "Has your organization implemented data loss prevention (DLP) to detect potential unauthorized access, use, or disclosure of client data?"
# query = input("What question would you like to ask? (Type \"STOP\" to exit): ")
# if query == "STOP":
#     break

prediction = pipe.run(query=query, params={"Retriever": {"top_k": 4}})

# Create prompt template
prompt = PromptTemplate(input_variables=["prefix", "question", "context"],
                        template="{prefix}\nQuestion: {question}\n Context: {context}\n")

# Provide instructions/prefix
prefix = "You are an assistant for the Information Security department of an enterprise designed to answer security questions in a professional manner. Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, confidence score, and answer'. Use the answers within the context to formulate your response in under two hundred words. In addition, list the referenced question IDs of the answers you referenced at the end of your response."

# Create context
context = ""
for answer in prediction["answers"]:
    context += "Question ID: {ID}, Confidence Score: {score}, Answer: {answer}\n".format(
        ID=answer.meta["question ID"], score = answer.score, answer=answer.meta["answer"])

# Generate Prompt
print("Generating prompt...")
prompt = prompt.format(prefix=prefix, question=query, context=context)
print("PROMPT:", prompt)

# Call openai API
response = openai.Completion.create(
    engine=deployment_name,
    prompt=(f"Question: {prompt}\n"
            "Answer:"
            ),
    max_tokens=1000,
    n=1,
    top_p=0.7,
    temperature=0.3,
    frequency_penalty=0.5,
    presence_penalty=0.2
)

gptResponse = response.choices[0].text.split('\n')[0]

print(f"OUTPUT:\n======================={gptResponse}")
