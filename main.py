import warnings
from tqdm.auto import tqdm  # progress bar
from datasets import load_dataset
import pandas as pd
import torch
import openai

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack import Document
from haystack.pipelines import FAQPipeline
from haystack.utils import print_answers

import langchain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

# Warning filter
warnings.filterwarnings('ignore', "TypedStorage is deprecated", UserWarning)

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
    "answer":
"""
   Yes, that is correct.     
""",
    "ci": "95.3%",
    "sources":
"""
* Source 1
* Source 2
* Source 3
"""
    },
    {
    "question": "Test 2 Question",
    "answer":
"""
    Test 2 answer.   
""",
    "ci": "50.3%",
    "sources":
"""
* Source 1
* Source 2
"""
    }
]

example_prompt = PromptTemplate(input_variables=["question", "answer", "ci", "sources"], template="Question: {question}\n Answer: {answer}\n Confidence Interval: {ci}\n Sources: {sources}")

# FewShotPrompt Template Generation
fs_prompt = FewShotPromptTemplate (
    examples=examples,
    example_prompt=example_prompt,
    suffix="""
Generate a coeherent response based off the following question using the above examples as formating reference. If the question cannot be answered with the information reply with 'Question cannot be answered.'
    Question: {question}\n
Please use information from the following context documents in the response and list the question IDs as sources in bullet points.
    Context: {context}
Also include the confience interval at the end of the answer.
    Confidence Interval: {ci}
    Answer:""",
    input_variables=["question", "context", "ci"]
)

# Normal Template
template = """Give a coherent response to the question based on the context below then include the confidence score and question IDs. Fill in the required information in the empty spaces below.

Question: {question}
Context: {context}
Confidence Score: {ci}
Question IDs: {ID}

Provde an answer then put confidence score after and put all question IDs as sources from the above information.

Output:
"""

gpt_prompt = PromptTemplate (
    input_variables=["context","question","ci","ID"],
    template=template
)

# Simple Template
template_simple = """Give a coherent and thorough response to the question based on the context below.

Question: {question}

Context: {context}

Answer:
"""

gpt_prompt_simple = PromptTemplate (
    input_variables=["question","context"],
    template=template_simple
)
print(f"Simple_Prompt: {gpt_prompt_simple}")

# Open AI Information
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
    prompt_ids = ""
    for answer in prediction["answers"]:
        # print(answer.meta["Question ID"])

        # Score calculations
        total_score += answer.score
        count += 1

        prompt_context += "Question ID: {ID}\n Content: {content}\n".format(ID=answer.meta["question ID"], content=answer.meta["answer"])
        prompt_ids += "{ID}\n".format(ID=answer.meta["question ID"])
    total_score /= count
    # print(f"Prompt Context:\n {prompt_context}")
    print(f"Mean Score: {total_score}")

    print("Generating prompt...")
    print(fs_prompt.format(question=prompt_question, context=prompt_context, ci=total_score))
    question = fs_prompt.format(question=prompt_question, context=prompt_context, ci=total_score)
    prompt_template = gpt_prompt.format(context=prompt_context, question=prompt_question,ci=total_score,ID=prompt_ids)
    prompt_template_simple = gpt_prompt_simple.format(question=prompt_question, context=prompt_context)

    # AI Response Prompt
    print("PROMPT:\n=======================",fs_prompt.format(question=prompt_question, context=prompt_context, ci=total_score), "\n=======================")
    response = openai.Completion.create(
        engine=deployment_name,
        prompt=(prompt_template_simple ),
        max_tokens=2000,
        n=1,
        top_p=0.7,
        temperature=0.7,
        frequency_penalty= 0.5,
        presence_penalty= 0.2
    )
    gptResponse = response.choices[0].text.split('\n')[0]
    print("OUTPUT:\n=======================",gptResponse,"\n=======================")

    # Manual output
    output = """
    Answer: {answer}
    Confidence Score: {ci}
    Sources:
{IDs}
    """

    output = output.format(answer=gptResponse, ci=total_score, IDs=prompt_ids)
    print(output)