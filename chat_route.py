from flask import request, jsonify
from modules import get_model, get_context_from_keywords, log_message, clean_reply

def chat_route(app):
    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json()
        user_message = data.get("message", "")
        model_key = data.get("model", "TinyLlama")

        agent, is_chat_model, has_tokenizer = get_model(model_key)
        print("The model:", model_key, flush=True)

        if agent is None:
            return jsonify({"error": f"Model '{model_key}' not found"}), 400

        log_message("user", user_message)
        context = get_context_from_keywords(user_message)

        if not context:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ]
            print("Messages sent to model:")
            for msg in messages:
                print(f"{msg['role'].capitalize()}: {msg['content']}", flush=True)

            if is_chat_model and has_tokenizer:
                prompt = agent.tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                prompt = user_message

            print("Final prompt sent to model:\n", prompt, flush=True)

            outputs = agent(prompt, max_new_tokens=128, do_sample=False, temperature=0.7, top_k=0, top_p=1.0)
            print("Raw model output:\n", outputs, flush=True)

            reply = outputs[0]["generated_text"]
            print("Assistant reply content:\n", reply, flush=True)

            reply = clean_reply(reply, user_message)
            log_message("assistant", reply)
            return jsonify({"response": reply})

        if context and len(context.split("\n")) == 1:
            keyword, response = context.split(": ", 1)
            print(f"Direct match found for keyword: {keyword}", flush=True)
            response = response.strip()
            if response.lower().startswith(keyword.lower()):
                response = response[len(keyword):].strip(" :.-")
            log_message("assistant", response)
            return jsonify({"response": response})

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{context}\n\n{user_message}" if context else user_message}
        ] if is_chat_model else None

        if is_chat_model and has_tokenizer:
            prompt = agent.tokenizer.apply_chat_template(messages, tokenize=False)
        elif not is_chat_model and has_tokenizer:
            prompt_text = f"{context}\n\nUser asked: {user_message}\nAnswer:" if context else user_message
            prompt = agent.tokenizer.decode(agent.tokenizer.encode(prompt_text, max_length=896, truncation=True))
        else:
            prompt = f"{context}\n\n{user_message}" if context else user_message

        print("Final prompt sent to model:\n", prompt, flush=True)

        outputs = agent(prompt, max_new_tokens=128, do_sample=False, temperature=0.7, top_k=0, top_p=1.0)
        print("Raw model output:\n", outputs, flush=True)

        reply = outputs[0]["generated_text"]
        reply = clean_reply(reply, user_message)
        log_message("assistant", reply)

        if is_chat_model:
            if "<|im_start|>assistant" in reply:
                reply = reply.split("<|im_start|>assistant")[1]
            elif "<|im_start|>user" in reply:
                reply = reply.split("<|im_start|>user")[-1]
            for tag in [
                "<|im_start|>", "<|im_end|>", "<||>", "〈||〉", "〈｜｜〉", "</s>", "**", "\\",
                "system", "user", "assistant", "header", "Chat History"
            ]:
                reply = reply.replace(tag, "")
            if user_message in reply:
                reply = reply.split(user_message)[-1]
            reply = reply.strip()

        print(f"Model reply:\n{reply}", flush=True)
        return jsonify({"response": reply})