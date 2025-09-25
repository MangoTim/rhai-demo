from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_agent import LlamaAgent
from gpt_agent import GPT
from qwen_agent import Qwen
import psycopg2
import re
from pdf_scanner import pdf_scanner

app = Flask(__name__)
CORS(app)

model_cache = {}
app.register_blueprint(pdf_scanner)

def get_db_connection():
    return psycopg2.connect(
        dbname="rhai_table",
        user="rhai",
        password="redhat",
        host="192.168.147.103",
        port=5432
    )

# def get_all_keywords():
#     conn = get_db_connection()
#     cur = conn.cursor()
#     cur.execute("SELECT keyword, response FROM keyword_responses")
#     results = cur.fetchall()
#     cur.close()
#     conn.close()
#     return {row[0].lower(): row[1] for row in results}

def get_model(model_key):
    if model_key == "TinyLlama":
        if "TinyLlama" not in model_cache:
            model_cache["TinyLlama"] = LlamaAgent()
        return model_cache["TinyLlama"], True
    elif model_key == "GPT-2":
        if "GPT-2" not in model_cache:
            model_cache["GPT-2"] = GPT()
        return model_cache["GPT-2"], False
    elif model_key == "Qwen":
        if "Qwen" not in model_cache:
            model_cache["Qwen"] = Qwen()
        return model_cache["Qwen"], True
    else:
        return LlamaAgent(), True  # fallback
    
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    model_key = data.get("model", "TinyLlama")

    # message_lower = user_message.lower()

    pipe, is_chat_model = get_model(model_key)

    def get_context_from_keywords(message):
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT keyword, response FROM keyword_responses")
            rows = cur.fetchall()
            cur.close()
            conn.close()

            message_lower = message.lower()
            context_parts = []
            matched_keywords = []

            rows.sort(key=lambda x: len(x[0]), reverse=True)

            for keyword, response in rows:
                # print(f"Checking keyword: {keyword}", flush=True)
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, message_lower):
                    context_parts.append(f"{keyword}: {response}")
                    matched_keywords.append(keyword)

            if not matched_keywords:
                print("No keywords matched for this message.", flush=True)

            print("Matched keywords:", matched_keywords, flush=True)
            print("Injected context from DB:\n", "\n".join(context_parts), flush=True)

            return "\n".join(context_parts) if context_parts else ""

        except Exception as e:
            print(f"Error in keyword matching: {e}", flush=True)
            return ""


    def log_message(role, message):
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO chat_history (role, message) VALUES (%s, %s)", (role, message))
        conn.commit()

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

    log_message("user", user_message)

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

    context = get_context_from_keywords(user_message)

    if context and len(context.split("\n")) == 1:
        keyword, response = context.split(": ", 1)
        print(f"Direct match found for keyword: {keyword}", flush=True)

        # Remove keyword from beginning of response if repeated
        response = response.strip()
        if response.lower().startswith(keyword.lower()):
            response = response[len(keyword):].strip(" :.-")

        # log_message("user", user_message)
        log_message("assistant", response)

        return jsonify({"response": response})

    if is_chat_model:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\n\n{user_message}" if context else user_message}
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)

    else:
        if context:
            prompt_text = f"{context}\n\nUser asked: {user_message}\nAnswer:"
            # prompt_text = f"{context}\n\nUser asked: {user_message}\nAnswer concisely and do not include additional questions or context."
        else:
            prompt_text = user_message

        prompt = pipe.tokenizer.decode(
            pipe.tokenizer.encode(prompt_text, max_length=896, truncation=True)
        )

    # history = get_recent_history()

    # if is_chat_model:
    #     messages = [{"role": "system", "content": "You are a helpful assistant."}]
    #     for role, msg in history:
    #         messages.append({"role": role, "content": msg})

    #     messages.append({"role": "user", "content": f"{context}\n\n{user_message}" if context else user_message})
    #     prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
    # else:
    #     history_text = "\n".join([f"{role.capitalize()}: {msg}" for role, msg in history])
    #     if context:
    #         prompt_text = f"{context}\n\n{history_text}\nUser asked: {user_message}\nAnswer:"
    #     else:
    #         prompt_text = f"{history_text}\nUser asked: {user_message}\nAnswer:"

    #     prompt = pipe.tokenizer.decode(
    #         pipe.tokenizer.encode(prompt_text, max_length=896, truncation=True)
    #     )

    print("Final prompt sent to model:\n", prompt, flush=True)


    outputs = pipe(
        prompt,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.7,
        top_k=0,
        top_p=1.0
    )


    # outputs = pipe(
    #     prompt,
    #     max_new_tokens=128,
    #     do_sample=False,
    #     temperature=0.7,
    #     top_k=0,
    #     top_p=1.0
    # )

    def clean_reply(reply, user_message):
        # Remove known formatting tokens
        for tag in [
            "<|im_start|>", "<|im_end|>", "<||>", "〈||〉", "〈｜｜〉", "</s>", "**", "\\",
            "system", "user", "assistant", "header", "Chat History"
        ]:
            reply = reply.replace(tag, "")

        # Remove repeated user input
        if user_message.strip().lower() in reply.strip().lower():
            reply = reply.lower().split(user_message.strip().lower())[-1].strip()

        return reply.strip()

    reply = outputs[0]["generated_text"]
    reply = clean_reply(reply, user_message)
    log_message("assistant", reply)

    if is_chat_model:
        if "<|im_start|>assistant" in reply:
            reply = reply.split("<|im_start|>assistant")[1]
        elif "<|im_start|>user" in reply:
            reply = reply.split("<|im_start|>user")[-1]

        # Remove known formatting tokens and clutter
        for tag in [
            "<|im_start|>", "<|im_end|>", "<||>", "〈||〉", "〈｜｜〉", "</s>", "**", "\\", 
            "system", "user", "assistant", "header", "Chat History"
        ]:
            reply = reply.replace(tag, "")

        # Remove repeated user input if present
        if user_message in reply:
            reply = reply.split(user_message)[-1]

        reply = reply.strip()

    print(f"Model reply:\n{reply}", flush=True)

    return jsonify({"response": reply})

@app.route("/history", methods=["GET"])
def get_history():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT role, message FROM chat_history
        WHERE role IN ('user', 'assistant')
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return jsonify([{"role": r, "message": m} for r, m in reversed(rows)])

if __name__ == "__main__":
    # model_cache["TinyLlama"] = LlamaAgent()
    # model_cache["GPT-2"] = GPT()
    # model_cache["Qwen"] = Qwen()
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)