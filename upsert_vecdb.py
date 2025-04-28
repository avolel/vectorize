import pandas as pd
import requests
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

#Initialize Pincone Client
pinecone = Pinecone(
        api_key=os.getenv('PINECONE_API_KEY')
        #host="us-east-1"        
    )
BATCH_SIZE = 100
METADATA_COLUMNS = ["Agency Name", "Work Location Borough", "Title Description","Base Salary","Pay Basis", "Regular Hours", 
                    "Regular Gross Paid", "OT Hours", "Total OT Paid", "Total Other Pay"]
index_name = "citpayroll"

#Load CSV file
df = pd.read_csv("Citywide_Payroll_Data__Fiscal_Year__20250425.csv")
texts = df["Fiscal Year"].fillna("").astype(str).tolist()
ids = ["item_" + str(i) for i in range(len(df))]

#Get (Ollama) embeddings
def get_ollama_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "input": text}
    )
    response.raise_for_status()
    data = response.json()
    #print(data)
    return data["embedding"]

# List existing indexes
existing_indexes = pinecone.list_indexes()

# Create index if it doesn't exist
if index_name not in existing_indexes:
    pinecone.create_index(
        index_name,
        dimension=768,  # Adjust based on your embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Modify if needed
    )

index = pinecone.Index(index_name)

#Upsert in batches of 100
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Upserting to Pinecone"):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_ids = ids[i:i + BATCH_SIZE]
    
    metadata_list = df[METADATA_COLUMNS].to_dict(orient="records")
    embeddings = [get_ollama_embedding(text) for text in batch_texts]

    vectors = [(id_, emb, metadata) for id_, emb, metadata in zip(batch_ids, embeddings, metadata_list)]
    index.upsert(vectors=vectors)