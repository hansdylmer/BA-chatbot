from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
import os
import chromadb
from whoosh.analysis import StemmingAnalyzer

schema = Schema(
    id=ID(stored=True, unique=True),
    title=TEXT(stored=True),
    heading=TEXT(stored=True),
    content=TEXT(stored=True)
)

if not os.path.exists("whoosh_index"):
    os.mkdir("whoosh_index")

ix = create_in("whoosh_index", schema)
writer = ix.writer()

client = chromadb.PersistentClient(path="./vector_database/embeddings_db_multilingual-e5-small")
collection = client.get_or_create_collection("SU_vector_database")
collection_data = collection.get()

for idx, text, meta in zip(collection_data["ids"], collection_data["documents"], collection_data["metadatas"]):
    writer.add_document(
        id=str(idx),
        title=meta["title"],
        heading=meta["heading"],
        content=text
    )

writer.commit()
print("✅ Whoosh index created.")
