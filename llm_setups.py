import os
from dataclasses import dataclass

# from llama_cpp import Llama
# import requests
#
# from pathlib import Path
# from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
import openai


from utils import ROOT_DIR


@dataclass
class LlamaHf:
    model: LlamaForCausalLM
    tokenizer: LlamaTokenizer
    label_to_token: dict


def setup_llama_hf(device: str, label_to_id: dict):
    model_id = 'huggyllama/llama-7b'

    tokenizer = LlamaTokenizer.from_pretrained(model_id, cache_dir="./models/llama-7b")
    model = LlamaForCausalLM.from_pretrained(model_id, cache_dir="./models/llama-7b")

    model = model.to(device)
    model.eval()

    # create label to token mapping
    label_to_token = {}
    for label in label_to_id.keys():
        token_id = tokenizer.encode(label)[1]
        label_to_token[label] = token_id

    return LlamaHf(model, tokenizer, label_to_token)


# def setup_llama(device: str, verbose: bool = False) -> Llama:
#     model_file_name = "llama-ggml-model-q4_0.bin"
#     local_path = f"./models/{model_file_name}"
#
#     Path(local_path).parent.mkdir(parents=True, exist_ok=True)
#
#     url = 'https://huggingface.co/TheBloke/LLaMa-7B-GGML/resolve/main/llama-7b.ggmlv3.q4_0.bin'
#
#     # Check if the model file already exists.
#     if not os.path.exists(local_path):
#         # Download the model file if it doesn't exist.
#         response = requests.get(url, stream=True)
#         with open(local_path, 'wb') as f:
#             for chunk in tqdm(response.iter_content(chunk_size=8192)):
#                 if chunk:
#                     f.write(chunk)
#     else:
#         # The model file already exists, so skip download.
#         print(f"Model file already exists at {local_path}. Skipping download.")
#
#     if device == 'cpu':
#         llm = Llama(model_path=local_path, logits_all=True, verbose=verbose)
#     else:
#         n_gpu_layers = 40
#         n_batch = 512
#         llm = Llama(model_path=local_path, n_gpu_layers=n_gpu_layers, n_batch=n_batch,
#                     logits_all=True, verbose=verbose)
#
#     return llm


def setup_openai_gpt3():
    # get OpenAI access key
    with open(os.path.join(ROOT_DIR, 'openai_key.txt'), 'r') as f:
        key = f.readline().strip()
        openai.api_key = key
