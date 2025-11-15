from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "google/flan-t5-base"  # You can replace this
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Your query and context from Chroma
user_query = "Hvordan søger jeg SU?"
context = "Du skal logge ind på su.dk med MitID. Der kan du finde ansøgningsskemaer og vejledning."

# Build prompt (instruct-tuned format)
prompt = f"[INST] Brug følgende kontekst til at besvare spørgsmålet: {context}\n\nSpørgsmål: {user_query} [/INST]"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)

# Decode and print result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
