from transformers import pipeline
import torch

def LlamaAgent():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,
        device_map="auto"
    )