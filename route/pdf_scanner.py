from flask import Blueprint, request, jsonify
import fitz  # PyMuPDF
import os
import glob
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer
from modules import get_db_connection, log_message, clean_reply, get_model

from model.llama_agent import LlamaAgent
from model.gpt_agent import GPTAgent
from model.qwen_agent import QwenAgent
from model.redhat_test import RedhatAgent
from model.deepseek_agent import DeepSeekAgent

from flasgger import swag_from
from api_docs.pdf_scanner_docs import upload_pdf_doc, ask_pdf_doc, remove_pdf_doc

pdf_scanner = Blueprint("pdf_scanner", __name__)
model_cache = {}

MODEL_TOKEN_LIMITS = {
    "tinyllama": 2048,
    "gpt-2": 1024,
    "qwen": 32000,
    "redhat": 4096,
    "deepseek": 4096
}

def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed file: {file_path}", flush=True)

@pdf_scanner.route("/upload_pdf", methods=["POST"])
@swag_from(upload_pdf_doc)
def upload_pdf():
    try:
        file = request.files.get("pdf")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        os.makedirs("uploads", exist_ok=True)
        clean_folder("uploads")

        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)
        print(f"File saved to: {file_path}", flush=True)

        text = extract_pdf_text(file_path)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM pdf_documents")
        cur.execute("""
            INSERT INTO pdf_documents (title, content, uploaded_at)
            VALUES (%s, %s, NOW())
        """, (filename, text))
        conn.commit()
        cur.close()
        conn.close()

        print(f"Inserted new PDF into database: {filename}", flush=True)
        return jsonify({"status": "scanned", "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@pdf_scanner.route("/ask_pdf", methods=["POST"])
@swag_from(ask_pdf_doc)
def ask_pdf():
    try:
        data = request.get_json()
        question = data.get("question")
        model_key = data.get("model", "TinyLlama")

        model, is_chat_model, has_tokenizer = get_model(model_key)
        if not model:
            return jsonify({"error": f"Unsupported model: {model_key}"}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT content FROM pdf_documents ORDER BY uploaded_at DESC LIMIT 1")
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return jsonify({"error": "No PDF found"}), 404

        pdf_text = row[0]

        # Token-aware truncation
        token_limit = MODEL_TOKEN_LIMITS.get(model_key.strip().lower(), 2048)
        reserved_tokens = 200
        max_pdf_tokens = token_limit - reserved_tokens

        if has_tokenizer and model_key.strip().lower() == "tinyllama":
            tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            encoded = tokenizer.encode(pdf_text, truncation=True, max_length=max_pdf_tokens)
            pdf_text = tokenizer.decode(encoded)

        prompt = (
            f"You are a helpful assistant. Answer the following question based on this document:\n"
            f"{pdf_text}\n\n"
            f"Question: {question}\n"
            f"Answer briefly (max 128 tokens):"
        )

        # print("Raw model prompt:\n", prompt, flush=True)

        result = model(prompt, max_new_tokens=128)
        answer = result[0]["generated_text"].strip()

        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()

        log_message("user", question)
        log_message("assistant", answer)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@pdf_scanner.route("/remove_pdf", methods=["DELETE"])
@swag_from(remove_pdf_doc)
def remove_pdf():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM pdf_documents")
        conn.commit()
        cur.close()
        conn.close()

        for file in glob.glob("uploads/*.pdf"):
            os.remove(file)

        return jsonify({"message": "PDF removed from database and uploads."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500