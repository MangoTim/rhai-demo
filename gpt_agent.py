from transformers import pipeline
import torch

def GPT():
    return pipeline(
        "text-generation",
        model="openai-community/gpt2",
        torch_dtype=torch.float32,
        device_map="auto"
    )