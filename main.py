from haystack.document_stores import FAISSDocumentStore
from datasets import load_dataset
import torch
from haystack.nodes import EmbeddingRetriever
from haystack import Document
from tqdm.auto import tqdm  # progress bar

from haystack.pipelines import FAQPipeline
import pandas as pd
from haystack.utils import print_answers

loaded = False
try:
    document_store = FAISSDocumentStore.load(index_path="my_faiss_index.faiss")
    loaded = True
except:
    document_store = FAISSDocumentStore(
        similarity="cosine",
        embedding_dim=768
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
    df["embedding"] = retriever.embed_queries(queries=questions).tolist()
    df = df.rename(columns={"question": "content"})

    # Convert Dataframe to list of dicts and index them in our DocumentStore
    docs_to_index = df.to_dict(orient="records")
    document_store.write_documents(docs_to_index)

    document_store.save(index_path="my_faiss_index.faiss")

print("docs:", document_store.get_document_count())
print("embeddings:", document_store.get_embedding_count())

pipe = FAQPipeline(retriever=retriever)

while True:

    query = input("What question would you like to ask? (Type \"STOP\" to exit): ")
    if query == "STOP":
        break
    
    prediction = pipe.run(query=query, params={"Retriever": {"top_k": 4}})

    print_answers(prediction, details="medium")

    # print(prediction["answers"][0].meta)

    for answer in prediction["answers"]:
        print(answer.meta["Question ID"])
