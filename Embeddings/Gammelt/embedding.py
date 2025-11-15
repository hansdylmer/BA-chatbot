import json
from langchain_community.document_loaders import JSONLoader
from pprint import pprint
from sentence_transformers import SentenceTransformer
import chromadb

# === 1. Load JSON document ===
file_path = "../Scraping/data/links_content_2025-08-04.json"
loader = JSONLoader(  # Loader hver paragraf som et dokument og tilføjer metadata.
    file_path=file_path,
    jq_schema=""".[] | 
        {link, title} as $meta |
        .sections[] | 
        {Origin: {link: $meta.link, title: $meta.title}, Heading: .heading, Content: .content}""",
    text_content=False
)
data = loader.load()

# print(data[1])

page_content = []
page_metadata = []
page_ID = []

for page_index, doc in enumerate(data):
    page_json = json.loads(doc.page_content)  # Nu er det et dict
    heading = page_json["Heading"]            # Nu kan du tilgå felter
    content = page_json["Content"]
    origin = page_json["Origin"]

    full_text = f"{heading}\n\n{content}"
    metadata = {
        "link": origin["link"],
        "title": origin["title"],
        "heading": heading
    }

    page_content.append(full_text)
    page_metadata.append(metadata)
    page_ID.append(page_index + 1)


model = SentenceTransformer("intfloat/multilingual-e5-small")
embeddings = model.encode(page_content)

client = chromadb.PersistentClient(path="./vector_database/seneste")
collection = client.get_or_create_collection("SU_vector_database")

collection.add(
    documents=page_content,
    embeddings=embeddings.tolist(),
    metadatas=page_metadata,
    ids=list(map(str, page_ID))
)

