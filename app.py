import os
openai.api_key = os.getenv("OPENAI_API_KEY")
client = QdrantClient(os.getenv("QDRANT_URL", "http://localhost:6333"))
from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import openai

app = FastAPI()

# Qdrant client setup (local ya tumhara hosted URL)
client = QdrantClient("http://localhost:6333")

# --- User Profile (Personalization) ---
user_profile = {
    "name": "Syed",
    "preferred_language": "English",  # ya "Urdu"
    "tone": "friendly",               # options: friendly, formal, concise
    "last_query": ""
}

# --- Ask Agent (Book Q&A with Qdrant + OpenAI) ---
@app.get("/ask")
def ask(query: str):
    embedding = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )["data"][0]["embedding"]

    search_result = client.search(
        collection_name="book_collection",
        query_vector=embedding,
        limit=3
    )

    context = " ".join([hit.payload["text"] for hit in search_result])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answering from the book."},
            {"role": "user", "content": f"Question: {query}\nContext: {context}"}
        ]
    )

    return {"answer": response["choices"][0]["message"]["content"]}


# --- Personalized Ask Agent ---
@app.get("/personalized_ask")
def personalized_ask(query: str):
    user_profile["last_query"] = query

    embedding = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )["data"][0]["embedding"]

    search_result = client.search(
        collection_name="book_collection",
        query_vector=embedding,
        limit=3
    )

    context = " ".join([hit.payload["text"] for hit in search_result])

    system_prompt = f"You are a {user_profile['tone']} assistant. Reply in {user_profile['preferred_language']}."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}\nContext: {context}"}
        ]
    )

    return {"answer": response["choices"][0]["message"]["content"]}


# --- Math Agent ---
@app.get("/math")
def math_solver(expression: str):
    try:
        result = eval(expression)
        return {"answer": f"Result: {result}"}
    except Exception as e:
        return {"error": str(e)}


# --- Translate Agent ---
@app.get("/translate")
def translate_to_urdu(text: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Translate the following text to Urdu"},
            {"role": "user", "content": text}
        ]
    )
    return {"translated": response['choices'][0]['message']['content']}


# --- Search Agent (placeholder) ---
@app.get("/search")
def search(query: str):
    return {"answer": f"Search results for: {query}"}
