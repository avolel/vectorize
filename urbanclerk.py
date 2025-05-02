from ollama import chat
from ollama import embed
from ollama import ChatResponse
from typing import List, Dict
from pinecone import Pinecone, PineconeApiException
import os
from dotenv import load_dotenv
import sys
import logging
from rich.markdown import Markdown
from rich.console import Console
from datetime import datetime

#Logging Setup
logging.basicConfig(
    filename='urbanclerk.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

load_dotenv()

PINECONEAPI_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = "citypayroll"

console = Console()

pinecone = Pinecone(
        api_key=PINECONEAPI_KEY
    )

index = pinecone.Index(INDEX_NAME)

system_prompt = {
    "role": "system",
    "content": (
        "You are a factual assistant who answers questions about New York City employees using only the provided context.\n\n"
        "If the context is missing, unrelated, or insufficient to answer the question with certainty, "
        "respond strictly with:\n\n"
        "\"I'm sorry, I don't have enough information to answer that based on the available data.\"\n\n"
        "Do not make up answers. Do not use outside knowledge. Only refer to the context provided in the conversation."
    )
}

#Add system prompt to chat history
chat_history: List[Dict[str, str]] = [system_prompt]

#RAG - get context from pinecone vector DB
def Retrieve_Context(query: str, top_k: int = 3) -> str:
    try:
        query_embedding = embed(model='nomic-embed-text', input=query)
        results = index.query(vector=query_embedding["embeddings"][0], top_k=top_k, include_metadata=True, namespace="nyc-city-payroll")

        if not results["matches"]:
            logging.warning("No relevant matches were found in Pinecone for this query: %s", query)
            return ""
        
        # Format each result into readable summary
        docs = []
        for match in results["matches"]:
            md = match["metadata"]
            summary = (
                f"Title: {md.get('Title Description')}\n"
                f"Location: {md.get('Work Location Borough')}\n"
                f"Base Salary: ${md.get('Base Salary'):,.2f} {md.get('Pay Basis')}\n"
                f"Gross Paid: ${md.get('Regular Gross Paid'):,.2f}\n"
                f"Other Pay: ${md.get('Total Other Pay'):,.2f}\n"
                f"OT Hours: {md.get('OT Hours')}, OT Pay: ${md.get('Total OT Paid'):,.2f}\n"
                f"Year: {int(md.get('Fiscal Year'))}"
            )
        docs.append(summary)
        
        return "\n\n---\n\n".join(docs)
    except PineconeApiException as e:
        logging.error("Pinecone API error: %s", str(e))
        return ""
    except Exception as e:
        logging.error("Unexpected error during context retrieval: %s", str(e))
        return ""

def LLamaChat(messages: List[Dict[str,str]]) -> ChatResponse:
    try:
        return chat(model='llama3.2', messages=messages,
            stream=True, options={"temperature": 0.0})
    except Exception as e:
        logging.error("Ollama chat failure: %s", str(e))
        return None

#Log Chat history to chat log
def FileLogger(user_input: str, assistant_reply: str):
    with open("chat_log.md", "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"### 🕒 {timestamp}\n")
        f.write(f"**You:**\n```\n{user_input.strip()}\n```\n\n")
        f.write(f"**Assistant:**\n{assistant_reply.strip()}\n\n---\n\n")

def Streaming(response: ChatResponse):
    assistant_response = ''

    if not response:
        print("Error generating response.")
        return ""
    
    try:
        for chunk in response:
            content = chunk["message"]["content"]
            assistant_response += content
    except Exception as e:
        logging.error("Streaming error: %s", str(e))
    return assistant_response

def Main():
    while True:
        try:
            prompt = input("\nYou: ").strip()
            if prompt.lower() == "exit":
                logging.info("Session ended by user.")
                sys.exit()
            
            logging.info("User prompt: %s", prompt)

            #RAG - receive context from pinecone
            context = Retrieve_Context(prompt)

            if not context:
                print("I'm sorry, I don't have enough information to answer that based on the available data.")
                continue

            #Inject the context into the convo
            augmented_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
            user_prompt = {'role': 'user', 'content': augmented_prompt}
            chat_history.append(user_prompt)

            response = LLamaChat(chat_history)
            reply = Streaming(response)
            print('\n')

            #Add Assistant response to chat history
            chat_history.append({'role': 'assistant', 'content': reply})
            console.print(Markdown(f"**Assistant:**\n{reply.strip()}\n"))
            FileLogger(prompt, reply)
        except KeyboardInterrupt as e:
            print("\nSession terminated.")
            break
        except Exception as e:
            logging.error("Unexpected failure in main loop: %s", str(e))
            print("Something went wrong. Check the logs.")

if __name__ == "__main__":
    Main()