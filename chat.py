import ollama
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

### Get vectorDB

client = PersistentClient(path="./Embeddings/vector_database/seneste")
collection = client.get_collection("SU_vector_database")

language_model = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"



# Get user input
input_query = input("Indtast dit spørgsmål: ")
embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
embedded_query = embedding_model.encode(input_query)
results = collection.query(
    query_embeddings=[embedded_query.tolist()],
    n_results=3,
    include=["documents", "metadatas", "embeddings", "distances"],
)

docs = results["documents"][0]
distances = results["distances"][0]

context_chunks = '\n'.join([f"- {doc}" for doc in docs])

instructions = f"""Du er en hjælpsom chatbot.
Brug udelukkende information fra den kontekst, du får, til at besvare spørgsmål.
Hvis du ikke kan finde informationen i konteksten, så sig 'Jeg ved det ikke'

{context_chunks}
"""



stream = ollama.chat(
    model=language_model,
    messages=[
        {"role": "system", "content": instructions},
        {"role": "user", "content": input_query},
    ],
    stream=True
)


print("Chatbot response:")
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)