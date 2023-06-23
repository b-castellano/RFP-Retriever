import os
import json
from haystack.document_stores import FAISSDocumentStore
import pandas as pb
from haystack import Document
from tqdm.auto import tqdm  # progress bar
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline

# Get configs
with open('configs.json') as user_file:
  configs = json.load(user_file)
  
# Create new local document store

newStore = False

if os.path.exists("/data/index.faiss"):
    document_store = FAISSDocumentStore.load(configs["index_path"])
    newStore = True
else:
    document_store = FAISSDocumentStore(faiss_index_factory_str=configs["faiss_index"])

# Define Retriever

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    model_format="sentence_transformers"
)

# Get QA data
data = pb.read_csv('qna.csv')


# Get dataframe with columns "question", "answer" and some custom metadata
    
data.fillna(value="", inplace=True)

# Create embeddings for our questions from the FAQs
# questions = list(data["question"].values)
# data["embedding"] = retriever.embed_queries(queries=questions).tolist()
# df = data.rename(columns={"question": "content"})

# Convert Dataframe to list of dicts and index them in our DocumentStore

docs_to_index = data.to_dict(orient="records")
document_store.write_documents(docs_to_index)

# Embed documents

document_store.update_embeddings(
   retriever,
   batch_size=128
)

# Save Store
if (newStore):

    document_store.save(index_path="data/index.faiss", config_path="data/config.json")


# Query QA pairs

# input = input("Enter a prompt or question:")
input = "How often is UHG audited?"

search_pipe = FAQPipeline(retriever)
result = search_pipe.run(
    query=input,
    params={"Retriever": {"top_k": 2}}
)

print_answers(result, details="medium")

# generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")


