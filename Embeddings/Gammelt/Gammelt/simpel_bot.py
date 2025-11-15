from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# 🔹 1. Indlæs dokument
with open("dit-fribeloeb-er-nedsat-naar-du-modtager-handicaptillaeg.txt", encoding="utf-8") as f:
    text = f.read()

# 🔹 2. Split dokument i bidder
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([text])

# 🔹 3. Embedding
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedder)
retriever = db.as_retriever()

# 🔹 4. LLM (lokal model)
model_name = "KennethTM/gpt2-medium-danish"  # eller en bedre dansk/multilingual model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
llm = HuggingFacePipeline(pipeline=pipe)

# 🔹 5. Stil et spørgsmål og hent kontekst manuelt
question = "Hvornår bliver fribeløbet nedsat?"
docs = retriever.get_relevant_documents(question)

# 🔹 6. Byg en dansk prompt manuelt
context = "\n\n".join([d.page_content for d in docs])
prompt = f"""Svar på spørgsmålet på baggrund af den følgende tekst. Hvis du ikke kan finde svaret, så sig "det fremgår ikke".

### Tekst:
{context}

### Spørgsmål:
{question}

### Svar:
"""

# 🔹 7. Generér svar
response = llm(prompt)
print("🧠 Svar:", response)
