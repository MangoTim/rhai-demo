from transformers import pipeline
import torch

def Qwen():
    """
    Loads and returns the Qwen2.5-1.5B-Instruct model pipeline.
    """
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="auto"
    )