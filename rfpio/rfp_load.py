import pandas as pd
import os
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import ExtractiveQAPipeline
from pathlib import Path


def init_store():
    try:

        rfp_store = FAISSDocumentStore.load(index_path=Path("rfpio/rfp_faiss_index.faiss").absolute())
        rfp_loaded = True
        print("\n\n=======================rfp data loaded====================\n\n")

    except:
        rfp_store = FAISSDocumentStore(
            sql_url="sqlite:///rfpio/rfp_document_store.db",
            similarity="cosine",
            embedding_dim=768,
            duplicate_documents='overwrite'
        )
        rfp_store.save(index_path="./rfpio/rfp_faiss_index.faiss")
        rfp_loaded = False
  
    return rfp_store, rfp_loaded

def parseQNAandEmbedDocuments(rfp_document_store, retriever):

    # Get dataframe with columns "question", "answer" and some custom metadata
    df = pd.read_csv("rfpio/qna.csv")
    df.fillna(value="", inplace=True)

    # Create embeddings for our questions from the FAQs
    questions = list(df["question"].values)
    print("questions:", len(questions))
    df["embedding"] = retriever.embed_queries(queries=questions).tolist()
    df = df.rename(columns={"question": "content"})

    # Convert Dataframe to list of dicts and index them in our DocumentStore
    docs_to_index = df.to_dict(orient="records")
    print("rfp documents:", len(docs_to_index))
    rfp_document_store.write_documents(docs_to_index)
    rfp_document_store.update_embeddings(retriever)

    rfp_document_store.save(index_path="./rfpio/rfp_faiss_index.faiss")
  
    print("rfp docs added:", rfp_document_store.get_document_count())
    print("rfp docs embedded:", rfp_document_store.get_embedding_count())
