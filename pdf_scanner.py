from flask import Blueprint, request, jsonify
import fitz # PyMuPDF
import os
import glob
from werkzeug.utils import secure_filename
from modules import get_db_connection, log_message
from model.llama_agent import LlamaAgent
from model.gpt_agent import GPTAgent
from model.qwen_agent import QwenAgent
from model.redhat_test import RedhatAgent
from model.deepseek_agent import DeepSeekAgent

pdf_scanner = Blueprint("pdf_scanner", __name__)
model_cache = {}

def get_model(model_key="TinyLlama"):
    if model_key not in model_cache:
        print(f"Initializing model: {model_key}", flush=True)
        if model_key == "GPT-2":
            model_cache[model_key] = GPTAgent()
        elif model_key == "Qwen":
            model_cache[model_key] = QwenAgent()
        # elif model_key == "DeepSeek":
        #     model_cache[model_key] = DeepSeekAgent()
        elif model_key == "RedHat":
            model_cache[model_key] = RedhatAgent()
        else:
            print(f"Unknown model_key '{model_key}', falling back to TinyLlama", flush=True)
            model_cache[model_key] = LlamaAgent()
    print(f"Using model: {model_key}", flush=True)
    return model_cache[model_key]

def extract_pdf_text(file_path, max_chars=10000):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) > max_chars:
            break
    return text

def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed file: {file_path}", flush=True)


@pdf_scanner.route("/upload_pdf", methods=["POST"])
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
def ask_pdf():
    try:
        data = request.get_json()
        question = data.get("question")
        model_key = data.get("model", "TinyLlama")

        supported_models = ["TinyLlama", "GPT-2", "Qwen", "RedHat"]
        if model_key not in supported_models:
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
        model = get_model(model_key)

        prompt = (
            f"You are a helpful assistant. Answer the following question based on this document:\n"
            f"{pdf_text}\n\n"
            f"Question: {question}\n"
            f"Answer briefly (max 128 tokens):"
        )

        print("Raw model output:\n", prompt, flush=True)

        result = model(prompt, max_new_tokens=128)
        answer = result[0]["generated_text"].strip()

        if answer.startswith(prompt):
            answer = answer[len(prompt):].strip()

        log_message("user", question)
        log_message("assistant", answer)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@pdf_scanner.route("/remove_pdf", methods=["POST"])
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