import json
from langchain_community.document_loaders import JSONLoader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
from pprint import pprint

# === 1. Load JSON document ===
file_path = "../Scraping/links_content_2025-08-01.json"
loader = JSONLoader( # Loader hver paragraf som et dokument og tilføjer metadata.
    file_path=file_path,
    jq_schema=""".[] | 
        {link, title} as $meta |
        .sections[] | 
        {Heading: .heading, Content: .content, Origin: {link: $meta.link, title: $meta.title}}""",
    text_content=False
)
data = loader.load()
    
pprint(data[:2])  # Print first two entries for verification

# === 2. Flatten content ===
documents = []
metadatas = []
ids = []



for page_index, doc in enumerate(data):
    page_data = json.loads(doc.page_content)
    heading = page_data.get("Heading", "")
    content = page_data["Content"]
    
    doc_text = f"{heading}\n\n{content}"  # 👈 Embed both
    
    doc_meta = {
        "title": page_data["Origin"]["title"],
        "link": page_data["Origin"]["link"],
        "heading": heading
    }
    
    doc_id = f"doc{page_index}"
    documents.append(doc_text)
    metadatas.append(doc_meta)
    ids.append(doc_id)


# === 3. Load multilingual model ===
model = SentenceTransformer("intfloat/multilingual-e5-small")
embeddings = model.encode(documents)

# === 4. Set up ChromaDB ===
client = chromadb.PersistentClient(path="./embeddings_db_multilingual-e5-small_inkl_foraeldre")
collection = client.get_or_create_collection("handi_og_foraeldre")

# === 5. Store in Chroma ===
collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=ids
)

print("✅ Embedding completed and stored in ChromaDB.")
