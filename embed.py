from ollama import embed

def get_ollama_embedding(text):
    response = embed(model='nomic-embed-text', input=text)
    return response["embeddings"]

data = get_ollama_embedding("Hello World")

print(data)