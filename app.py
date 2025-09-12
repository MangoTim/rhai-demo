from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_agent import LlamaAgent
from gpt_agent import GPT
from qwen_agent import Qwen

app = Flask(__name__)
CORS(app)

model_cache = {}

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

    pipe, is_chat_model = get_model(model_key)

    print("Selected model:", model_key)

    if is_chat_model:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        prompt = user_message

    outputs = pipe(
        prompt,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.7,
        top_k=0,
        top_p=1.0
    )

    reply = outputs[0]["generated_text"]

    if is_chat_model:
        if "<|im_start|>assistant" in reply:
            reply = reply.split("<|im_start|>assistant")[1]
        elif "<|im_start|>user" in reply:
            reply = reply.split("<|im_start|>user")[-1]

        # Remove known formatting tokens and clutter
        for tag in [
            "<|im_start|>", "<|im_end|>", "<||>", "</s>", "<||>",
            "system", "user", "assistant", "header", "Chat History"
        ]:
            reply = reply.replace(tag, "")

        # Remove repeated user input if present
        if user_message in reply:
            reply = reply.split(user_message)[-1]

        reply = reply.strip()

    return jsonify({"response": reply})

if __name__ == "__main__":
    # model_cache["TinyLlama"] = LlamaAgent()
    # model_cache["GPT-2"] = GPT()
    # model_cache["Qwen"] = Qwen()
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)