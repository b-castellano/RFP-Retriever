import os
from haystack.document_stores import FAISSDocumentStore
import pandas as pb
from haystack import Document
from tqdm.auto import tqdm  # progress bar
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline


# Create new local document store

document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")



# Get QA data
data = pb.read_csv('qna.csv')


# Add documents to DB

batchSize = 2000

dicts = []

for row in data.index:
    
    # create haystack document object with text content and doc metadata
    if row % batchSize == 0:
        document_store.write_documents(dicts)
        dicts.clear()
 
    dicts.append({
    'id': data["Question ID"][row],
    'content': data["Answer"][row],
    'meta': {
        "SME": data["SME"][row],
        "Question": data["Question"][row],
        "Alternate": data["Alternate Questions"][row]
    }
})

document_store.write_documents(dicts)


# Embed documents

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    model_format="sentence_transformers"
)
document_store.update_embeddings(
   retriever,
   batch_size=128
)

# Query QA pairs

# input = input("Enter a prompt or question:")
# # input = "How often is UHG audited?"

# search_pipe = FAQPipeline(retriever)
# result = search_pipe.run(
#     query=input,
#     params={"Retriever": {"top_k": 2}}
# )

document_store.save(index_path="data/index.faiss", config_path="data/config.json")\


# Load document store
document_store = FAISSDocumentStore.load(index_path="data/index.faiss", config_path="data/config.json")

print("docs: ",document_store.get_document_count())
print("embeds: ", document_store.get_embedding_count())

# print_answers(result, details="medium")

# generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")


