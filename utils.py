import os
from copy import deepcopy
from typing import Tuple

import numpy as np
import pickle


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunk_size_helper(params: dict) -> int:
    # Set the batch size. Default to 1
    # Useful when calling a LLM API with a batch of prompts
    bs = params['batch_size']
    if bs is None:
        bs = 1

    return bs


def random_sampling(sentences: list[str], labels: list[int], num: int, seed: int = 0) -> Tuple[list[str], list[int]]:
    """Randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"number of samples [{num}] is more than the total size of the pool {len(labels)}, can't perform random sampling"

    np.random.seed(seed)

    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]

    return deepcopy(selected_sentences), deepcopy(selected_labels)


def load_experiment_data_pickle(params: dict) -> dict:
    # load saved results
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"

    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")

    return data


def save_experiment_data_pickle(params: dict, data: dict) -> dict:
    # save results
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")

    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")

    return data


def load_results(params_list: list[dict]):
    # load saved results
    raise NotImplementedError