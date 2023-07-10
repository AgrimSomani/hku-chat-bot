import openai
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import pinecone
import streamlit as st

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(OPENAI_API_KEY)

openai.api_key= OPENAI_API_KEY

model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="us-west4-gcp-free"
)
index_name = "hku-chatbot"
index = pinecone.Index(index_name)

def get_metadata_text(result):
    to_return = []
    for i in range(len(result['matches'])):
        to_return.append(result['matches'][i]['metadata']['text'])
    return "\n".join(to_return)


def find_match(input):
    input_embedding = model.encode(input).tolist()
    result = index.query(input_embedding,top_k=10,include_metadata=True)
    
    return get_metadata_text(result)

def query_refiner(conversation,query):

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):

        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


