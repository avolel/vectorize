import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import os
from ollama import embed
from dotenv import load_dotenv

load_dotenv()
PINECONEAPI_KEY = os.getenv('PINECONE_API_KEY')

#Initialize Pincone Client
pinecone = Pinecone(
        api_key=PINECONEAPI_KEY
    )
BATCH_SIZE = 500
METADATA_COLUMNS = ["Fiscal Year", "Work Location Borough", 
                    "Title Description","Base Salary","Pay Basis", "Regular Hours", 
                    "Regular Gross Paid", "OT Hours", "Total OT Paid", "Total Other Pay"]
index_name = "citypayroll"

#Load CSV file
df = pd.read_csv("Citywide_Payroll_Data__Fiscal_Year__20250425.csv")
texts = df["Agency Name"].fillna("").astype(str).tolist()
ids = [str(i) for i in range(len(df))]

#Get (Ollama) embeddings
def get_ollama_embedding(text):
    response = embed(model='nomic-embed-text', input=text)
    return response["embeddings"][0]

index = pinecone.Index(index_name)

#Upsert in batches of 500
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Upserting to Pinecone", ncols=100):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_ids = ids[i:i + BATCH_SIZE]
    batch_metadata = df.iloc[i:i + BATCH_SIZE][METADATA_COLUMNS].to_dict(orient="records")

    embeddings = [get_ollama_embedding(text) for text in batch_texts]

    vectors = [(id_, emb, metadata) for id_, emb, metadata in zip(batch_ids, embeddings, batch_metadata)]
    index.upsert(vectors=vectors, namespace="nyc-city-payroll")