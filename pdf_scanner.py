from flask import Blueprint, request, jsonify
import fitz
import psycopg2
import os
import glob
from llama_agent import LlamaAgent
from gpt_agent import GPT
from qwen_agent import Qwen

pdf_scanner = Blueprint("pdf_scanner", __name__)
model_cache = {}

def get_model(model_key="TinyLlama"):
    if model_key == "TinyLlama":
        if "TinyLlama" not in model_cache:
            model_cache["TinyLlama"] = LlamaAgent()
        return model_cache["TinyLlama"]
    elif model_key == "GPT-2":
        if "GPT-2" not in model_cache:
            model_cache["GPT-2"] = GPT()
        return model_cache["GPT-2"]
    elif model_key == "Qwen":
        if "Qwen" not in model_cache:
            model_cache["Qwen"] = Qwen()
        return model_cache["Qwen"]
    else:
        return LlamaAgent()  # fallback

def get_db_connection():
    return psycopg2.connect(
        dbname="rhai_table",
        user="rhai",
        password="redhat",
        host="192.168.147.103",
        port=5432
    )

def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)

def log_message(role, message):
    conn = get_db_connection()
    cur = conn.cursor()

    # Insert new message
    cur.execute("INSERT INTO chat_history (role, message) VALUES (%s, %s)", (role, message))
    conn.commit()

    # Keep only latest 10 messages (5 pairs)
    cur.execute("""
        DELETE FROM chat_history
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT id
                FROM chat_history
                WHERE role IN ('user', 'assistant')
                ORDER BY timestamp DESC
                LIMIT 10
            ) AS latest_pairs
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed file: {file_path}", flush=True)

@pdf_scanner.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    file = request.files.get("pdf")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    os.makedirs("uploads", exist_ok=True)

    # for old_file in glob.glob("uploads/*.pdf"):
    #     os.remove(old_file)
    #     print(f"Removed old file: {old_file}", flush=True)

    clean_folder("uploads")

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    print(f"File saved to: {file_path}", flush=True)

    text = extract_pdf_text(file_path)

    # Clean up database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM pdf_documents")
    print("PDF history has been cleaned up.", flush=True)

    # Insert new PDF (no summary column)
    cur.execute("""
        INSERT INTO pdf_documents (title, content, uploaded_at)
        VALUES (%s, %s, NOW())
    """, (file.filename, text))
    conn.commit()
    print(f"Inserted new PDF into database: {file.filename}", flush=True)

    cur.close()
    conn.close()

    return jsonify({"status": "scanned", "filename": file.filename})

@pdf_scanner.route("/ask_pdf", methods=["POST"])
def ask_pdf():
    data = request.get_json()
    question = data.get("question")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT content FROM pdf_documents ORDER BY uploaded_at DESC LIMIT 1")
    row = cur.fetchone()
    cur.close()
    conn.close()

    def get_recent_history(limit=5):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT role, message FROM chat_history
            WHERE role IN ('user', 'assistant')
            ORDER BY timestamp DESC
            LIMIT %s
        """, (limit * 2,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return list(reversed(rows))  # Oldest first

    if not row:
        return jsonify({"error": "No PDF found"}), 404

    pdf_text = row[0]
    model = get_model()

    prompt = f"You are a helpful assistant. Answer the following question based on this document:\n{pdf_text[:3000]}\n\nQuestion: {question}\nAnswer briefly (max 128 tokens):"
    # prompt = (
    #     f"You are a helpful assistant. Based on the following document, answer the user's question.\n\n"
    #     f"{pdf_text}\n\n"
    #     f"{question}\n\n"
    #     f"Respond clearly and concisely."
    # )
    # prompt = f"Answer the following question based only on this document:\n{pdf_text[:3000]}\n\nQuestion: {question}\nAnswer concisely and do not include additional questions or context. Max 128 tokens."
    history = get_recent_history(limit=5)
    history_text = "\n".join([f"{role.capitalize()}: {msg}" for role, msg in history])
    # prompt = (
    #     f"You are a helpful assistant. Based on the following document and recent conversation, answer the user's next question.\n\n"
    #     f"{pdf_text[:3000]}\n\n"
    #     f"{history_text}\n\n"
    #     f"{question}"
    # )
    result = model(prompt, max_new_tokens=128)
    answer = result[0]["generated_text"].strip()

    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()

    # Log both user question and assistant answer
    log_message("user", question)
    log_message("assistant", answer)

    # print("Final prompt sent to model:\n", prompt, flush=True)

    return jsonify({"answer": answer})

@pdf_scanner.route("/remove_pdf", methods=["POST"])
def remove_pdf():
    # Delete from DB
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM pdf_documents")
    conn.commit()
    cur.close()
    conn.close()

    # Delete uploaded files
    for file in glob.glob("uploads/*.pdf"):
        os.remove(file)

    return jsonify({"message": "PDF removed from database and uploads."})