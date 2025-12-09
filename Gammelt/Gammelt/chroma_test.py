from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import numpy as np
from numpy.linalg import norm

# Setup til embedding af Query
model = SentenceTransformer("intfloat/multilingual-e5-small")
client = PersistentClient(path="./vector_database/seneste")
collection = client.get_collection("SU_vector_database")




# Step 1: Query embedding
query_text = input("Indtast din forespørgsel: ")
query_vec = model.encode(query_text)

# Step 2: Get top matches (with embeddings)
results = collection.query(
    query_embeddings=[query_vec.tolist()],
    n_results=3,
    include=["documents", "metadatas", "embeddings", "distances"],
    
)

print(f"Query: {query_text}\nResults: {results}")

# all_docs = collection.get(include=["documents", "embeddings", "metadatas"])
# # Step 3: Compute cosine similarity manually
# def cosine_sim(a, b):
#     return np.dot(a, b) / (norm(a) * norm(b))

# scored = []
# for doc, embed, meta in zip(all_docs["documents"], all_docs["embeddings"], all_docs["metadatas"]):
#     score = cosine_sim(query_vec, embed)
#     scored.append((score, doc, meta))

# # 5. Sortér stigende (dårligst først)
# scored.sort(key=lambda x: x[0])  # laveste cosine similarity først

# # 6. Print de 5 dårligste
# print("\n💀 De 5 dårligste matches:\n")
# for score, doc, meta in scored[:250]:
#     print(f"🎯 Similarity: {score:.4f}\n📄 {doc[:200]}\n🔗 {meta.get('link')}\n")

# for doc, meta, embed in zip(results["documents"][0], results["metadatas"][0], results["embeddings"][0]):
#     score = cosine_sim(query_vec, embed)
#     print(f"\n📄 {doc[:200]}\n\n🎯 Similarity: {score:.4f}")
