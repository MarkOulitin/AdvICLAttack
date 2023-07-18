from typing import Tuple

import numpy as np
import openai

# from llama_cpp import Llama
# from llm_setups import setup_llama

from llm_setups import LlamaHf, setup_llama_hf
from utils import chunks, chunk_size_helper


def construct_prompt(params: dict,
                     train_sentences: list[str],
                     train_labels: list[int],
                     test_sentence: str) -> str:
    """construct a single prompt to be fed into the model"""

    assert params['task_format'] == 'classification'

    # take the prompt template and fill in the training and test example
    prompt = params['prompt_prefix']
    q_prefix = params['q_prefix']
    a_prefix = params['a_prefix']
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix + s + "\n"
        prompt += a_prefix + params['label_dict'][l][0] + "\n\n"

    prompt += q_prefix + test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1]  # GPT models do not want a trailing space, so we cut off -1

    return prompt


# def create_completion_llama(prompt: str,
#                             label_to_id: dict,
#                             max_tokens: int,
#                             temperature: float = 0.0,
#                             num_logprobs: int = None,
#                             stop: str = "\n",
#                             echo=False,
#                             llm: Llama = None,
#                             device: str = 'cpu') -> Tuple[dict, np.array]:
#     if llm is None:
#         llm = setup_llama(device)
#
#     response = llm(prompt, max_tokens=max_tokens, stop=[stop], logprobs=num_logprobs,
#                    temperature=temperature, echo=echo)
#
#     llm_response = response['choices'][0]
#     label_probs = get_probs_llama_cpp(label_to_id, len(label_to_id.keys()), llm_response)
#
#     return llm_response, label_probs


def create_completion_llama_hf(prompt: str,
                               label_to_id: dict,
                               max_tokens: int,
                               device: str,
                               llm_hf: LlamaHf,
                               temperature: float = 0.0) -> dict:
    tokenizer = llm_hf.tokenizer
    model = llm_hf.model
    label_to_token = llm_hf.label_to_token

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation_output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        output_scores=True,
        return_dict=True,
        return_dict_in_generate=True,
        temperature=temperature
    )

    # print(generation_output)
    sequence = generation_output['sequences'][0]
    sequence_decoded = tokenizer.decode(sequence)

    score = generation_output['scores'][0][0]
    # print(score)

    num_classes = len(label_to_token.keys())
    probs = [0] * num_classes

    for label, label_id in label_to_id.items():
        label_token = label_to_token[label]
        label_score = score[label_token].cpu()

        probs[label_id] += np.exp(label_score)

    probs = np.array(probs)
    probs = probs / np.sum(probs)  # normalize to 1

    # Check if the sum of probabilities is close to 1.0
    tolerance = 1e-6
    is_close_to_one = np.isclose(probs.sum(), 1.0, rtol=tolerance)
    assert is_close_to_one

    # print(sequence_decoded)
    # print(probs)
    return sequence_decoded, probs


def create_completion_gpt3(prompt: str,
                           max_tokens: int,
                           temperature: float = 0.0,
                           num_logprobs: int = None,
                           stop: str = "\n",
                           echo=False,
                           n=None) -> dict:
    return openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=num_logprobs,
        stop=stop,
        echo=echo,
        n=n
    )


def create_completion(prompt: str,
                      label_to_id: dict,
                      max_tokens: int,
                      model_name: str,
                      device: str,
                      llm,
                      temperature: float = 0.0,
                      num_logprobs: int = None,
                      stop: str = "\n",
                      echo: bool = False,
                      n: int = None):
    """Complete the prompt using a language model"""

    assert max_tokens >= 0
    assert temperature >= 0.0

    if 'llama_hf' in model_name:
        return create_completion_llama_hf(prompt, label_to_id, max_tokens, device, llm, temperature)

    elif 'llama_cpp' in model_name:
        raise NotImplementedError
        # return create_completion_llama_cpp(prompt, labels, max_tokens, temperature, num_logprobs, stop, echo,
        #                                     device=device, llm=llm)
    elif 'gpt3' in model_name:
        return create_completion_gpt3(prompt, max_tokens, temperature, num_logprobs, stop, echo, n)
    else:
        raise NotImplementedError


# def get_model_response(params: dict,
#                        train_sentences: list[str],
#                        train_labels: list[int],
#                        test_sentences: list[str],
#                        num_tokens_to_predict_override: int = None) -> Tuple[list[dict], list[str]]:
#     """
#     Get model's responses on test sentences, given the training examples
#     :param params: parameters for the experiment
#     :param train_sentences: few-shot training sentences
#     :param train_labels: few-shot training labels
#     :param test_sentences: few-shot test sentences
#     :param num_tokens_to_predict_override: whether to override num token to predict
#     :return: a tuple containing list of responses dictionaries and prompts
#     """
#
#     all_raw_answers = []
#
#     prompts = [construct_prompt(params, train_sentences, train_labels, test_sentence) for test_sentence in
#                test_sentences]
#
#     chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
#     for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
#         if num_tokens_to_predict_override is not None:
#             num_tokens_to_predict = num_tokens_to_predict_override
#         else:
#             num_tokens_to_predict = params['num_tokens_to_predict']
#         resp = create_completion(test_chunk_prompts, params['inv_label_dict'], num_tokens_to_predict, params['model'],
#                                  num_logprobs=params['api_num_logprob'])
#         for answer_id, answer in enumerate(resp['choices']):
#             all_raw_answers.append(answer)
#
#     return all_raw_answers, prompts


def get_probs_llama_cpp(label_to_id: dict,
                        num_classes: int,
                        llm_response: dict):
    top_logprobs = llm_response['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token
    probs = [0] * num_classes

    for label, label_id in label_to_id.items():
        label = " " + label  # notice prompt does not have space after 'A:'
        if label in top_logprobs:
            probs[label_id] += np.exp(top_logprobs[label])

    probs = np.array(probs)
    probs = probs / np.sum(probs)  # normalize to 1

    return probs


def get_all_probs(params: dict, raw_resp: list[dict], test_sentences: list[str]) -> np.array:
    """Obtain model's label probability for each of the test examples. The returned prob is normalized"""
    assert len(raw_resp) == len(test_sentences)

    num_classes = len(params['label_dict'].keys())

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    for i, llm_response in enumerate(raw_resp):
        label_probs = get_probs(params, num_classes, llm_response)
        all_label_probs.append(label_probs)

    all_label_probs = np.array(all_label_probs)  # probs are normalized

    return all_label_probs
