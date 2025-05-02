from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from tqdm import tqdm 

load_dotenv()
PINECONEAPI_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = "citypayroll"

#Chunk - Init RecursiveCharacterTextSplitter object
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50, 
    add_start_index=True
)

#Embedding - Init OllamaEmbeddings object
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

#Initialize Pincone Client
pinecone = Pinecone( api_key=PINECONEAPI_KEY )
index = pinecone.Index(INDEX_NAME)

vector_store = PineconeVectorStore(embedding=embeddings_model, index=index, namespace="nyc-city-payroll")

#Load CSV
file_path = "Citywide_Payroll_Data__Fiscal_Year__20250425.csv"
loader = CSVLoader(file_path=file_path, encoding="utf-8")
data = loader.load()

#Split Docs
all_splits = text_splitter.split_documents(data)

def batch_documents(documents, batch_size=500):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

batches = list(batch_documents(all_splits, batch_size=100))

for batch in tqdm(batches, desc="Upsert to Pinecone"):
    vector_store.add_documents(documents=batch)