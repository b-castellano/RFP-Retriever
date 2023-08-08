from turtle import pd
from langchain.document_loaders import PyPDFLoader
import os
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from tqdm import tqdm
import re

from haystack import Document



def getAllFileNames(directory):
    return os.listdir(directory)

def init_store():
    try:
        sp_store = FAISSDocumentStore.load(index_path="./sharepoint/sp_faiss_index.faiss")
        sp_loaded = True
    except:
        sp_store = FAISSDocumentStore(
            sql_url="sqlite:///sharepoint/sp_document_store.db",
            similarity="cosine",
            embedding_dim=768,
            duplicate_documents='overwrite'
        )
        sp_loaded = False

    return sp_store, sp_loaded

def parseFilesAndEmbedDocuments(directory, filenames, sp_document_store, sp_retriever):
    
    docs = []
    i = 0
    for filename in tqdm(filenames, desc="Processing PDF Files: "):
        path = fr"{directory}{filename}"

        type = path.split(".")[-1]

        # We are only processing PDF files
        if type == "pdf":
            try:

                pdfLoader = PyPDFLoader(path)
                pages = pdfLoader.load_and_split()
            except:
                continue

            for page in pages:
          
                text = re.sub(r"[\n,\t,•]", "", page.page_content)
                
                id = i

                doc = Document(
                    content= text,
                    id= id,
                    meta = {

                        "filename" : filename,
                        "page" : page.metadata['page']

                    }
                )

                docs.append(doc)
                i+=1
                

    print("sharepoint documents to add to DB:", len(docs))

    sp_document_store.write_documents(docs)
    sp_document_store.update_embeddings(sp_retriever)

    print("sharepoint docs added:", sp_document_store.get_document_count())
    print("sharepoint docs embedded:", sp_document_store.get_embedding_count())

    sp_document_store.save(index_path="./sharepoint/sp_faiss_index.faiss")


def main():

    directory = r"/Users/dhoule5/OneDrive - UHG/EIS Artifacts/"


    filenames = getAllFileNames(directory)

    print(len(parseFilesAndEmbedDocuments(directory,filenames)))
    
if __name__ == "__main__": 
    main()