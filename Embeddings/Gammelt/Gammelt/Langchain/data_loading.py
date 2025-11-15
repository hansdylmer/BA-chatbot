from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
import json
from pathlib import Path
from huggingface_hub import login
login("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")

file_path = "../Scraping/links_content2025-07-30.json"
data = json.loads(Path(file_path).read_text(encoding="utf-8")) #Printer hvert subsite som en document


loader = JSONLoader( # Loader hver paragraf som et dokument og tilføjer metadata.
    file_path=file_path,
    jq_schema=""".[] | 
        {link, title} as $meta |
        .sections[] | 
        {Heading: .heading, Content: .content, Origin: {link: $meta.link, title: $meta.title}}""",
    text_content=False
)
documents = loader.load()

document_content_to_embed = [doc.page_content for doc in documents]
# print(f"JSONLoader output: {documents}\n")
# print(f"json.loads: output: {type(data[1])}")

### Text splitting
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100,
#     separators=["\n\n", "\n", " ", ""]
# )

# final_documents = []
# for doc in documents:
#     if len(doc.page_content) > 1000:
#         split_docs = splitter.split_text(doc.page_content)
#         final_documents.extend(split_docs)
#     else:
#         final_documents.append(doc.page_content)
        
# for i, doc in enumerate(final_documents):
#     print(type(doc))
#     if isinstance(doc, str):
#         print(f"⚠️ Item {i} is a string! Expected Document.")

# print(len(documents), len(final_documents))
# texts = [doc for doc in final_documents]
# print(len(texts))

# # print(type(final_documents), type(final_documents[1].page_content))

### Forskellige embedding modeller som kan bruges:
model_name = "intfloat/multilingual-e5-small"
# model_name = "BAAI/bge-small-en-v1.5"
# model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
    # cache_folder=Path("./embeddings_cache") / model_name.replace("/", "_")
)



### Initializing Chroma vector store
vectorstore = Chroma(
    collection_name="handicaptillaeg",
    embedding_function=hf,
    persist_directory="./embeddings_db_multilingual-e5-small"
)


# vectorstore.add_documents(documents=documents) ## fjern kun, når der er nye dokumenter
vectorstore.add_documents(documents=documents, ids=[f"doc_{i}" for i in range(len(documents))])

query = "Hvilke ting tager I højde for, når I vurderer om jeg kan få handicaptillæg?"

query_embedding = hf.embed_query(query)
results = vectorstore.similarity_search_by_vector(query_embedding, k=5)

retrieved = vectorstore._collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["embeddings", "documents", "metadatas"]
)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

query_vec = np.array(query_embedding).reshape(1, -1)  # Reshape to 2D array for cosine similarity
doc_vecs = np.array(retrieved["embeddings"])

print(query_vec.shape, doc_vecs.shape)

similarities = cosine_similarity(query_vec, doc_vecs[0])[0]
distances = 1 - similarities

for i, (doc, dist) in enumerate(zip(retrieved["documents"], distances)):
    print(f"\n\nRank {i+1} | Distance: {dist:.4f} | Document: {doc[:100]}... ")

# for i, doc in enumerate(results):
#     print(f"\n--- Resultat {i+1} ---")
#     print(f"Indhold:\n{doc.page_content}")
#     print(f"Metadata:\n{doc.metadata}")
#     # Find distance if available
#     distance = doc.metadata.get("distance", "N/A")
#     print(f"Distance: {distance}")
    

