from email import header
from turtle import pd
from langchain.document_loaders import PyPDFLoader
import os
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from tqdm import tqdm
import re
from langchain.text_splitter import CharacterTextSplitter
import pdfplumber

import PyPDF4
import fitz
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
    doc_num = 0
    for filename in tqdm(filenames, desc="Processing PDF Files: "):
        path = fr"{directory}{filename}"

        type = path.split(".")[-1]
        print("==================\n",filename)

        # We are only processing PDF files
        if type == "pdf":
            all_text = test_splitter(path)
            if all_text is None:
                continue
           
            for text in all_text:
                
                if len(text) == 1:
                    header = ""
                    doc_content = text[0]
                else:
                    header = text[0]
                    doc_content = text[0].strip() + " - " + text[1]
                
                if doc_content.strip() == "" or len(re.findall("[/,:]", doc_content)) > 30 or len(doc_content) < 50:
                    continue
                
                # content = re.sub(r"[\n,\t,•]", "", doc_content)
                doc = Document(
                    content= doc_content,
                    id= doc_num,
                    meta = {

                        "filename" : filename,
                        "header" : header
                    }
                )
                print("Doc: " + doc.content)
                docs.append(doc)
                # print("CONTENT: ", doc.content)
                # docs.append(doc)
                doc_num +=1
            print("=================")

    print("sharepoint documents to add to DB:", len(docs))

    sp_document_store.write_documents(docs)
    sp_document_store.update_embeddings(sp_retriever)

    print("sharepoint docs added:", sp_document_store.get_document_count())
    print("sharepoint docs embedded:", sp_document_store.get_embedding_count())

    sp_document_store.save(index_path="./sharepoint/sp_faiss_index.faiss")


def test_splitter(file_path):
    
    try:
        with pdfplumber.open(file_path) as pdf: 

            all_text = []
         
            for page in pdf.pages:
                
                text_type = []
                
                bold_str = ""
                norm_str = ""

                isBolded = False
                
                try:
                    page.objects['char']
                except:
                    continue
                for char in page.objects['char']:
                    
                    if char["object_type"] == "char" and "Bold" in char["fontname"]:
                        if (not isBolded):
                            text_type.append(norm_str)
                            norm_str = ""
                            all_text.append(text_type)
                            text_type = []
                            isBolded = True
                        bold_str += char['text']

                    elif char["object_type"] == "char" and "Bold" not in char["fontname"]:
                        if (isBolded):
                            text_type.append(bold_str)
                            bold_str = ""
                            isBolded = False
                        norm_str += char['text']

                text_type.append(norm_str)
                #print("----------------\n", text_type, "\n---------------")
                all_text.append(text_type)

        return all_text

    except:

        return None

def main():

    test_splitter(r"/Users/dhoule5/OneDrive - UHG/EIS Artifacts/UnitedHealth Group - File Transfer - Electronic Communication Gateway (ECG).pdf")

    # directory = r"/Users/dhoule5/OneDrive - UHG/EIS Artifacts/"


    # filenames = getAllFileNames(directory)

    # print(len(parseFilesAndEmbedDocuments(directory,filenames)))
    
if __name__ == "__main__": 
    main()




# def parseFilesAndEmbedDocuments(directory, filenames, sp_document_store, sp_retriever):
    
#     docs = []
#     i = 0
#     for filename in tqdm(filenames, desc="Processing PDF Files: "):
#         path = fr"{directory}{filename}"

#         type = path.split(".")[-1]

#         # We are only processing PDF files
#         if type == "pdf":
#             try:

#                 pdfLoader = PyPDFLoader(path)
#                 pages = pdfLoader.load_and_split()
#             except:
#                 continue

#             for page in pages:
          
#                 text = re.sub(r"[\n,\t,•]", "", page.page_content)
                
#                 id = i

#                 doc = Document(
#                     content= text,
#                     id= id,
#                     meta = {

#                         "filename" : filename,
#                         "page" : page.metadata['page']

#                     }
#                 )

#                 docs.append(doc)
#                 i+=1
                

#     print("sharepoint documents to add to DB:", len(docs))

#     sp_document_store.write_documents(docs)
#     sp_document_store.update_embeddings(sp_retriever)

#     print("sharepoint docs added:", sp_document_store.get_document_count())
#     print("sharepoint docs embedded:", sp_document_store.get_embedding_count())

#     sp_document_store.save(index_path="./sharepoint/sp_faiss_index.faiss")




