import os
from langchain.llms import LlamaCpp
import openai
import requests

from pathlib import Path
from tqdm import tqdm
from utils import ROOT_DIR


def setup_llama() -> LlamaCpp:
    model_file_name = "llama-ggml-model-q4_0.bin"
    local_path = f"./models/{model_file_name}"

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    url = 'https://huggingface.co/TheBloke/LLaMa-7B-GGML/resolve/main/llama-7b.ggmlv3.q4_0.bin'

    # Check if the model file already exists.
    if not os.path.exists(local_path):
        # Download the model file if it doesn't exist.
        response = requests.get(url, stream=True)
        with open(local_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)
    else:
        # The model file already exists, so skip download.
        print(f"Model file already exists at {local_path}. Skipping download.")

    n_gpu_layers = 40
    n_batch = 256
    llm = LlamaCpp(model_path=local_path, n_gpu_layers=n_gpu_layers, n_batch=n_batch,
                   logits_all=True, verbose=True)

    return llm


def setup_openai_gpt3():
    # get OpenAI access key
    with open(os.path.join(ROOT_DIR, 'openai_key.txt'), 'r') as f:
        key = f.readline().strip()
        openai.api_key = key
