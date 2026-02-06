import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

label = "NAME_OF_YOUR_TRAINING_LABEL"
dataset = "harry_potter" # harry_potter / newsqa / procon / ragtruth
chunk_size = 128
K = 3
retriever = "bge-small-en-v1.5" # bge-small-en-v1.5 / all-MiniLM-L6-v2 / contriever / e5-base-v2
model = "Qwen3-8B" # Llama-3-8B-Instruct / Qwen3-8B
LLM_PATH = f"PATH_TO_YOUR_GENERATOR"
device = "cuda:0"
EMBED_PATH = f"PATH_TO_YOUR_EMBEDDING_MODEL"
DATA_DIR = f"PATH_TO_YOUR_DATA_FOLDER"
IO_PATH = f"PATH_TO_YOUR_INPUT_JSON_FILE"
OUT_DIR = f"PATH_TO_YOUR_OUTPUT_FILE"
INDEX_PATH = f"PATH_TO_YOUR_INDEX_FOLDER"

def load_all_documents(root_path=DATA_DIR):
    documents = []
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if fname.endswith(".txt"):
                loader = TextLoader(os.path.join(dirpath, fname), encoding="utf-8")
                documents.extend(loader.load())
    return documents

def token_length(text):
    return len(embed_tokenizer.encode(text, add_special_tokens=False))

def split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_size/2):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) # Counting by character numbers
    # splitter = RecursiveCharacterTextSplitter(                                                    # Counting by token numbers
    #     chunk_size=chunk_size,
    #     chunk_overlap=int(chunk_overlap),
    #     length_function=token_length,
    #     separators=["\n\n", "\n", ". ", " ", ""]
    # )
    return splitter.split_documents(documents)

def build_faiss_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_PATH)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(INDEX_PATH)
    return db

class LocalLLMWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, context, question):
        prompt = f"Consider the information below as context:\n{context}\nAnswer the following question: {question}"
        messages = [
            {"role": "system", "content": "You are a helpful language assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(**model_inputs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

def rag_answer(query, db, llm):
    retriever = db.as_retriever(search_kwargs={"k": K})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = llm.predict(context, query)
    return {
        "answer": answer,
        "retrieved_docs": [doc.page_content for doc in docs]
    }

if os.path.exists(INDEX_PATH):
    print(f"FAISS index found at {INDEX_PATH}, loading directly...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_PATH)
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vectorstore loaded. Skipping document loading and splitting.")
else:
    print("FAISS index not found. Processing documents...")
    print("Loading documents...")
    raw_docs = load_all_documents()
    print("Splitting documents...")
    split_docs = split_documents(raw_docs)
    print("Building FAISS vectorstore...")
    db = build_faiss_vectorstore(split_docs)

print("Loading local model & embedder...")
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_PATH)
model = AutoModelForCausalLM.from_pretrained(LLM_PATH, torch_dtype="auto", device_map=None).to(device)
llm_wrapper = LocalLLMWrapper(model, tokenizer)

print("Reading input JSON and processing queries...")
with open(IO_PATH, "r", encoding="utf-8") as f:
    items = json.load(f)

os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)

with open(OUT_DIR, "w", encoding="utf-8") as f_out:
    f_out.write("[\n")
    for idx, item in enumerate(tqdm(items, desc="Processing queries")):
        query_id = item["id"]
        query = item["question"]
        correct_answer = item["correct answer"]
        incorrect_answer = item["incorrect answer"]
        result = rag_answer(query, db, llm_wrapper)
        output = {
            "id": query_id,
            "input": query,
            "lm_output": result["answer"],
            "retrieved_docs_str": "\n\n".join(result["retrieved_docs"]),
            "correct answer": correct_answer,
            "incorrect answer": incorrect_answer
        }
        json.dump(output, f_out, ensure_ascii=False, indent=2)
        if idx < len(items) - 1:
            f_out.write(",\n")
        else:
            f_out.write("\n")
    f_out.write("]\n")

print(f"All results saved to {OUT_DIR}")
