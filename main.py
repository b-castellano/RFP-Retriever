import pandas as pd
from datasets import load_dataset
import traceback
import re

import openai

from haystack.nodes import EmbeddingRetriever
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers


import langchain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

def init():
     # Initialize document store
    document_store, loaded = init_store()

    # Initialize retriever
    retriever = init_retriever(document_store)

    # If new store, add documents and embeddings
    if not loaded:
        write_docs(document_store, retriever)
    
    # Initialize GPT
    init_gpt()

    # Initialize pipeline for document search
    return init_pipe(retriever)

def init_store():
    try:

        return FAISSDocumentStore.load(index_path="my_faiss_index.faiss"), True
        
    except:
        return FAISSDocumentStore(
            similarity="cosine",
            embedding_dim=768,
            duplicate_documents='overwrite'
        ), False

def init_retriever(document_store):
    return EmbeddingRetriever(
        document_store=document_store,
        embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
        model_format="sentence_transformers"
    )

def write_docs(document_store, retriever):
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
    print("docs added:", document_store.get_document_count())
    print("docs embedded:", document_store.get_embedding_count())


def init_pipe(retriever):
    return FAQPipeline(retriever=retriever)

def get_response(pipe, query):

    prediction = query_faiss(query, pipe)
        
    # Generate prompt from related docs
    prompt,scores, alts = create_prompt(query, prediction)

    # Feed prompt into gpt
    return call_gpt(prompt, scores, alts)

def query_faiss(query, pipe):
    # while True:
    # query = input("What question would you like to ask? (Type \"STOP\" to exit): ")
    # if query == "STOP":
    #     break
    return pipe.run(query=query, params={"Retriever": {"top_k": 5}})


# Create prompt template
def create_prompt(query, prediction):

    prompt = PromptTemplate(input_variables=["prefix", "question", "context"],
                            template="{prefix}\nQuestion: {question}\n Context: {context}\n")

    # Provide instructions/prefix
    prefix = """You are an assistant for the Information Security department of an enterprise designed to answer security questions professionally. Provided is the original question and some context consisting of a sequence of answers in the form of 'question ID, answer'. Use the answers within the context to answer the original question in a concise manner with explanation. List the question IDs of the answers you referenced to formulate your response."""

    # Create context
    context = ""
    scores = {}
    alts = []
    count = 0
    for answer in prediction["answers"]:
        newAnswer = re.sub("[\[\]'\"]","",answer.meta["answer"])
        # Remove docs 
        context += "Question ID: {ID}, Answer: {answer}\n".format(
            ID=answer.meta["cid"], answer=newAnswer)
        
        # Add ID-Score pair to dict
        scores[answer.meta["cid"]] = answer.score

        if (count < 3):
            
            alts.append(answer.meta["cid"])
            count+=1
    return prompt.format(prefix=prefix, question=query, context=context), scores, alts
    


def init_gpt():


    openai.api_key = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6Ii1LSTNROW5OUjdiUm9meG1lWm9YcWJIWkdldyIsImtpZCI6Ii1LSTNROW5OUjdiUm9meG1lWm9YcWJIWkdldyJ9.eyJhdWQiOiJodHRwczovL2NvZ25pdGl2ZXNlcnZpY2VzLmF6dXJlLmNvbSIsImlzcyI6Imh0dHBzOi8vc3RzLndpbmRvd3MubmV0L2RiMDVmYWNhLWM4MmEtNGI5ZC1iOWM1LTBmNjRiNjc1NTQyMS8iLCJpYXQiOjE2ODk5NjUyMzYsIm5iZiI6MTY4OTk2NTIzNiwiZXhwIjoxNjg5OTcwNTM2LCJhY3IiOiIxIiwiYWlvIjoiQVZRQXEvOFRBQUFBaEVyS2VqYkdBTWVEUkQ4UXBmd2k4UysrUjBEWUl6ejlIQklQS3dURURMbzlCeTlZeHJGc1ljb3ppVHkrdm4yeU82NUJ0MnYyWVJEL3F0YWpyYWZvTGxGSHQ0YUQ5ZXp2NzM2T0VOdFhxemM9IiwiYW1yIjpbInB3ZCIsIndpYSIsIm1mYSJdLCJhcHBpZCI6IjE4YTY2ZjVmLWRiZGYtNGMxNy05ZGQ3LTE2MzQ3MTJhOWNiZSIsImFwcGlkYWNyIjoiMiIsImZhbWlseV9uYW1lIjoiSG91bGUiLCJnaXZlbl9uYW1lIjoiRHlsYW4iLCJncm91cHMiOlsiM2ZkMmM4MGQtYzVkYS00NTI0LTgxMzQtYmY4YzEyNDc5MjMxIiwiOTYwMmJhMWUtZjc5OS00Yzg4LTkxNmEtMTRhYjZjNmI1YjlhIiwiZjVhNmRkMjUtODZjOS00OGMzLWJiMjYtNTczODJlYzNhMWE4IiwiZjQzYzNjMmQtZjkwZC00OWIyLWI2OTgtYTdhYjViMGJkYWRlIiwiZGM1NTc2MzctNGQ0OS00YmNhLTk1ZmEtMDI5Yjg5MDI2NWYyIiwiM2E1ODFjNGItZmZmNC00NGIyLTgxNGUtMzhmMzlhZThmN2JmIiwiOTFlYzI5NGItNTEzMS00ZDRlLTk2ZmQtZTI5OTI1OTJiZmViIiwiYjlkZWY4NTEtOGE2NS00ODVkLTljOWQtNTI0ZDdhMzc4YmU5IiwiMDJlNzc4NjItYWM3OC00ZDJkLTliMTEtMTA1OTk5OWNkZDg0IiwiYzYzMDYzNjctODAyNC00ZDdkLTljM2ItMjZlZTIxZmJhZjM2IiwiNzgzZjEzNjktYWFkYy00YTc0LTg4N2MtYjA4ZWM2MmQzOTVmIiwiMTE1MTk0NmMtNDA5NS00NGMzLWIyZDYtMDMyNmFiYjgzZWZlIiwiYWMxMmMxNmMtZWEzNy00YWU5LWE0NmItNTIzM2M4NjE4MmVlIiwiZTFlMmQxNzAtYWFlOS00NDFlLWFmZDUtYjkxYzBmM2JhZGU2IiwiMzk3M2VlOGQtYmExNy00ZWIxLTlmMWUtZDJiODY1YzYzMmNiIiwiYzUwOGViOTEtZTVkNC00YzA1LTljMWMtNzk2NjBiZjYxNTJiIiwiY2ZkMzBhOTItM2I5ZC00YWUwLWE5NDMtNjQwMDA0NDZiZTkzIiwiNjZlOWI5OWItNjFkMC00MmI3LTg1YjYtYTE3Y2Q3MGYyNjdiIiwiY2UwNzNiOWYtOWMyOC00MWRhLTk3Y2EtOTVlMjViMmY2ODc5IiwiNzA3YTc4YWUtZTAwOC00YTI1LTllNTctODA5NzM4ZGVlZTdmIiwiNDI5M2YwYmItZDA4ZC00ZGViLWFhY2YtYWQ0NTQ4NTBlZTZhIiwiZDUyZWRjY2QtOWMwMS00ZTIxLWE3YmEtNzc4MjQ0NDhlZGQzIiwiOGFkYjQ1Y2UtN2U0ZC00NWFjLWI2NDUtOTlkNDdkNTMzMDk0IiwiN2MxM2I0ZGYtOGZlZS00ZjUwLTkyNTItMThlNGE4ZjNjODgzIiwiMTFkMGIyZjUtNjFmMy00NzRlLWEyYmItZDZjYmI2MTQ1MWM5IiwiNjkyZjM1ZjYtZjZjZS00MmNhLWFmNDAtYjU2MWFkMzZhZDBlIiwiNDQ1ZWYwZjktZjkwMi00OWMzLTg4MGEtNDAyNTI0MjA2ZTRhIl0sImlwYWRkciI6IjE2OC4xODMuMTM1LjI0IiwibmFtZSI6IkhvdWxlLCBEeWxhbiBOIiwib2lkIjoiZTE1OTIwZWUtZjRlNC00MTg2LThjZWEtMWY0ZjRjM2UzMzY4Iiwib25wcmVtX3NpZCI6IlMtMS01LTIxLTU4ODM3MTU4My0xODA1MjY1NDYwLTQyMTQxMzMwNzMtNzU5MzE0MiIsInB1aWQiOiIxMDAzMjAwMjk4QjhERDlDIiwicmgiOiIwLkFSc0F5dm9GMnlySW5VdTV4UTlrdG5WVUlaQWlNWDNJS0R4SG9PMk9VM1NiYlcwYkFOWS4iLCJzY3AiOiJ1c2VyX2ltcGVyc29uYXRpb24iLCJzdWIiOiJ1OHltU1V5YnZlbDJfVEw3S01OYWdKSXFFeGhJZUVJYjluUTR3d3YxbWs0IiwidGlkIjoiZGIwNWZhY2EtYzgyYS00YjlkLWI5YzUtMGY2NGI2NzU1NDIxIiwidW5pcXVlX25hbWUiOiJkeWxhbl9ob3VsZUBvcHR1bS5jb20iLCJ1cG4iOiJkeWxhbl9ob3VsZUBvcHR1bS5jb20iLCJ1dGkiOiJESG4xUnRNdWxFYWE2Y09jdVRXV0FBIiwidmVyIjoiMS4wIiwid2lkcyI6WyJiNzlmYmY0ZC0zZWY5LTQ2ODktODE0My03NmIxOTRlODU1MDkiXX0.J7beC4ZNDJuL_2JCBjBzLKIcOOvVyMEQSc4h7V6UhIcn_FNmtqZ8AOdrn4MVCTS65m4YLNIzkiMv_9PGHojc8Hzo0E7a0je8mRwemE18WTVZfpiYm7jgdnBAZULZF8MOaIWzX9rflczskQvhXeouewHshA1or5vxsioSigmY_rAm3Zqdc95vpJUUMx9_uiwvjOe7LDYfqGmxjZSjii14fYll0KwLD56IYl-vhX_7P-SSa4Tr7N-UkWzEO_zca5W8_YwOMJYVLWhWlBUFWv3xnoQwNXmI2IwXRrJyo9sLiEhFcNNZIAj2_5wG31H_Pk3w2z6IYhC3TGM9yVzevfXkrg"
    openai.api_type = "azure_ad"
    openai.api_version = "2023-03-15-preview"
    openai.api_base = f"https://ays6pb1ntgslwvyopenai.openai.azure.com/"
  
    


    # openai.api_key = "dd9d2682f30f4f66b5a2d3f32fb6c917"
    # openai.api_type = "azure"
    # openai.api_version = "2023-06-01-preview"
    # openai.api_base = "https://immerse.openai.azure.com/"
    

# Call openai API
def call_gpt(prompt,scores,alts):

    deployment_id = "deployment-e86654ae68004190a12be5a187df27db"
    response = openai.Completion.create(
        engine=deployment_id,
        prompt=(f"Original Question: {prompt}\n"
                "Answer:"
                ),
        max_tokens=500,
        n=1,
        top_p=0.7,
        temperature=0.3,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    output = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
    
    print(output)
    ids = re.findall("CID\d+", output)
    ids = list(set(ids))
    output = re.sub("\(?(CID\d+),?\)?", "", output)

    # Handle case where gpt doesn't output sources in prompt
    if ids is None or len(ids) == 0:
        alternates = ""
        for i in alts:
            alternates += f"{i.strip()}\n"
        return f"{output}\nHere are some possible sources to reference:\n{alternates}"

    confidence = compute_average(ids,scores)
    output = output[ 0 : output.rindex(".") + 1]
   
    output += f"\nConfidence Score: {confidence:.2f}%"
    output += f"\nSources:\n"
    for i in ids:
        output += f"{i.strip()}\n"

    return output

def compute_average(ids, scores):

    total = 0

    for id in ids:

        id = id.strip()
        total += scores[id]

    avgscore = total / len(ids)     # convert total score to avg
    avgscore *= 100                 # convert from decimal to percentage

    return avgscore


def main():

    try:

        # Initialize FAISS store and create pipe instance
        pipe= init()

        while(True):
            
            # Users question
            query = input("Please ask a question. Reply 'STOP' to stop:")

            if query == "STOP":
                break

            output = get_response(pipe, query)
            print(f"OUTPUT:\n{output}")
        

       

    except:
        
        print("Error initializing var")
        traceback.print_exc()

if __name__ == "__main__":
    main()