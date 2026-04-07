"""Test multiple HF models to find which ones work with the free token."""
import os, json
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

hf_key = os.getenv("HF_TOKEN", "") or os.getenv("OPENAI_API_KEY", "")

models = [
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]

client = OpenAI(api_key=hf_key, base_url="https://router.huggingface.co/v1")
results = {}

for m in models:
    try:
        r = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            max_tokens=5,
            timeout=15,
        )
        results[m] = "OK"
    except Exception as e:
        results[m] = str(e)[:120]

with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to test_results.json")
