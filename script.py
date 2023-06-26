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

if os.path.exists(configs["index_path"]):

    document_store = FAISSDocumentStore.load(index_path=configs["index_path"], config_path=configs["config_path"])
    
else:
    document_store = FAISSDocumentStore(faiss_index_factory_str=configs["faiss_index"])
    newStore = True

# Define retriever

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    model_format="sentence_transformers"
)

# Add documents to index
if (newStore):
    
    data = pb.read_csv('qna.csv')
    data.fillna(value="", inplace=True)

    batchSize = 256
    docs = []

    for row in data.index:

        doc = Document(
            id=row,
            content= data["answer"][row],
            meta={
                "SME": data["sme"][row],
                "Question": data["question"][row],
                "Alternate": data["alternate questions"][row],
                "ID":data["question id"][row]
            }
        )
        docs.append(doc)
        # create haystack document object with text content and doc metadata
        if  row != 0 and (row % batchSize == 0 or row == len(data.index) - 1):
            document_store.write_documents(docs)
            embeds = retriever.embed_documents(docs)
            for i, doc in enumerate(docs):
                doc.embedding = embeds[i]
            document_store.write_documents(docs)
            print(row, " / ", len(data.index), " files added")
            docs.clear()

    # Save Document Store
    document_store.save(index_path=configs["index_path"], config_path=configs["config_path"])



# Embed documents

document_store.update_embeddings(
   retriever,
   batch_size=128
)

# Query 

# input = input("Enter a prompt or question:")
input = "How often is UHG audited?"

search_pipe = FAQPipeline(retriever)

result = search_pipe.run(
    query=input,
    params={"Retriever": {"top_k": 2}}
)

print("doc count:", document_store.get_document_count())

print_answers(result, details="medium")

# generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")


