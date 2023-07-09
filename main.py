from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

directory = './data'

# function to load documents
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

# fucntion to split document content into chunks
def split_docs(documents,chunk_size=500,overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=overlap)
    docs = text_splitter.split_documents(documents)
    return docs


# documents = load_docs(directory)
# docs = split_docs(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="us-west4-gcp-free"
)

index_name = "hku-chatbot"
index = Pinecone.from_existing_index(index_name=index_name,embedding=embeddings)

# function for getting similar docs from pinecone db using semantic search, based on input query
def get_similar_docs(query,k=10,score=False):
        similar_docs = index.similarity_search_with_score(query,k=k) if score else index.similarity_search(query,k=k)

        return similar_docs

query = "Honours"

similar_docs = get_similar_docs(query,score=True)

print(similar_docs)
