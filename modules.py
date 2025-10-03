import psycopg2
import re
from model.llama_agent import LlamaAgent
from model.gpt_agent import GPTAgent
from model.qwen_agent import QwenAgent
from model.redhat_test import RedhatAgent
from model.deepseek_agent import DeepSeekAgent

model_cache = {}

def get_db_connection():
    return psycopg2.connect(
        dbname="rhai_table",
        user="rhai",
        password="redhat",
        host="192.168.147.103",
        port=5432
    )

def get_model(model_key):
    key = model_key.strip().lower()

    if key == "tinyllama":
        if "tinyllama" not in model_cache:
            model_cache["tinyllama"] = LlamaAgent()
        return model_cache["tinyllama"], True, True

    elif key == "gpt-2":
        if "gpt-2" not in model_cache:
            model_cache["gpt-2"] = GPTAgent()
        return model_cache["gpt-2"], False, True

    elif key == "qwen":
        if "qwen" not in model_cache:
            model_cache["qwen"] = QwenAgent()
        return model_cache["qwen"], True, True

    elif key == "deepseek":
        if "deepseek" not in model_cache:
            model_cache["deepseek"] = DeepSeekAgent()
        return model_cache["deepseek"], True, False

    elif key == "redhat":
        if "redhat" not in model_cache:
            model_cache["redhat"] = RedhatAgent()
        return model_cache["redhat"], True, False

    print(f"Unknown model key: {model_key}", flush=True)
    return None, False, False


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

def clean_reply(reply, user_message):
    for tag in [
        "<|im_start|>", "<|im_end|>", "<||>", "〈||〉", "〈｜｜〉", "</s>", "**", "<|user|>", "<|assistant|>",
        "system", "user", "assistant", "header", "Chat History"
    ]:
        reply = reply.replace(tag, "")
    if len(user_message.strip()) > 5 and user_message.strip().lower() in reply.strip().lower():
        reply = reply.lower().split(user_message.strip().lower())[-1].strip()
    return reply.strip()