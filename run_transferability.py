import argparse
from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd
import textattack
import torch

from attack_dataset import ICLTransferabilityDataset
from attack_utils import ICLModelWrapper
from data_utils import load_dataset
from icl_attaker import ICLAttacker
from llm_setups import setup_llama_hf
from transferability_attack import ICLTransferabilityAttack
from utils import random_sampling


def main(models: list[str],
         datasets: list[str],
         seeds: list[int],
         num_few_shots: list[int],
         asr_experiment_csv_file_name: str,
         subsample_test_set: int,
         api_num_logprob: int):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        'subsample_test_set': subsample_test_set,
        'api_num_logprob': api_num_logprob,
        'asr_experiment_csv_file_name': asr_experiment_csv_file_name
    }

    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for num_shots in num_few_shots:
                for seed in seeds:
                    p = deepcopy(default_params)
                    p['model'] = model
                    p['dataset'] = dataset
                    p['seed'] = seed
                    p['num_shots'] = num_shots
                    p['expr_name'] = f"transferability_{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                    all_params.append(p)


    run_experiments(all_params)


def run_experiments(params_list):
    """
    Run the experiments and save its responses and the rest of configs
    """

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    print(f"device={device}")
    print(f"# of gpus={torch.cuda.device_count()}")

    for param_index, params in enumerate(params_list):
        experiment_name = params['expr_name']
        print("\nExperiment name:", experiment_name)

        # load data
        example_sentences, example_labels, test_sentences, test_labels = load_dataset(params)
        print(f"dataset {params['dataset']} stats:")
        print(f"size of examples pool: {len(example_labels)}")
        print(f"num of test sentences: {len(test_labels)}")

        # llm setup
        llm = setup_llama_hf(device, params['inv_label_dict'])

        # sanity check
        experiment_sanity_check(params, device, llm)

        # set seed
        np.random.seed(params['seed'])

        # sample test set
        if params['subsample_test_set'] is None:
            print(f"selecting full test set ({len(test_labels)} examples)")
            test_sentences, test_labels = test_sentences, test_labels
        else:
            print(f"selecting {params['subsample_test_set']} subsample of test set")

            test_sentences, test_labels = random_sampling(test_sentences,
                                                          test_labels,
                                                          params['subsample_test_set'])

        attack, attack_dataset, attack_args = setup_transferability_attack_experiment(llm,
                                                                                      device,
                                                                                      test_sentences,
                                                                                      test_labels,
                                                                                      example_sentences,
                                                                                      example_labels,
                                                                                      experiment_name,
                                                                                      params)


        attacker = ICLTransferabilityAttack(experiment_name, attack, attack_dataset, attack_args)
        attacker.attack_dataset()


def setup_transferability_attack_experiment(llm,
                                            device,
                                            test_sentences,
                                            test_labels,
                                            example_sentences,
                                            example_labels,
                                            experiment_name,
                                            params):
    def extract_adversarial_demonstrations(csv_file_name: str, top_k: int = 3) -> Tuple[list[str], list[str]]:
        df = pd.read_csv(csv_file_name)

        # Filter rows where result_type is 'Successful'
        filtered_df = df[df['result_type'] == 'Successful']

        # Calculate the difference between perturbed_score and original_score
        filtered_df['score_difference'] = filtered_df['perturbed_score'] - filtered_df['original_score']

        # Sort by score_difference in descending order and select the top 10 rows
        top_k_rows = filtered_df.nlargest(top_k, 'score_difference')

        original_texts = top_k_rows['original_text'].tolist()
        perturbed_texts = top_k_rows['perturbed_text'].tolist()

        return original_texts, perturbed_texts

    icl_model_wrapper = ICLModelWrapper(llm, device)
    attack = ICLTransferabilityAttack.build(icl_model_wrapper)

    # extract top 3 adversarial demonstrations
    original_texts, adversarial_demonstrations = extract_adversarial_demonstrations(params['asr_experiment_csv_file_name'])

    for original_demonstration, adversarial_demonstration in zip(original_texts, adversarial_demonstrations):
        # search adversarial demonstration index in the examples pool
        adversarial_demonstration_index = example_sentences.index(original_demonstration)
        experiment_name += "demonstration" + str(adversarial_demonstration_index)  # include demonstration index in experiment's name

        attack_dataset = ICLTransferabilityDataset(original_demonstration,
                                                   adversarial_demonstration,
                                                   adversarial_demonstration_index,
                                                   test_sentences, test_labels,
                                                   example_sentences, example_labels,
                                                   params['num_shots'], params)

        attack_args = textattack.AttackArgs(
            num_examples=len(attack_dataset),
            log_to_txt=f"./log_{experiment_name}.txt",
            log_to_csv=f"./log_{experiment_name}.csv",
            log_summary_to_json=f"./attack_summary_log_{experiment_name}.json",
            csv_coloring_style="plain",
            checkpoint_interval=5,
            checkpoint_dir="./checkpoints",
            disable_stdout=True,
        )

        return attack, attack_dataset, attack_args


def experiment_sanity_check(params: dict, device: str, llm):
    """Sanity check the experiment"""

    assert params['num_tokens_to_predict'] == 1

    model_name = params['model']

    # for classification, make sure that all of the class names are one word.
    for key, label_names in params['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            prompt = label_name

            if model_name == 'llama_hf':
                label_to_token = llm.label_to_token
                assert label_name in label_to_token

            else:
                raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True,
                        help='name of model(s), e.g., Llama')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True,
                        help='name of dataset(s), e.g., sst2')
    parser.add_argument('--seeds', dest='seeds', action='store', required=True,
                        help='seeds for the training set')
    parser.add_argument('--num_few_shots', dest='num_few_shots', action='store', required=True,
                        help='num training examples to use')
    parser.add_argument('--asr_experiment_csv_file_name', dest='asr_experiment_csv_file_name', action='store',
                        required=True, help='name of asr experiment csv file to load demonstrations')
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_logprob', dest='api_num_logprob', action='store', required=False, type=int,
                        default=100,
                        help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')

    args = parser.parse_args()
    args = vars(args)


    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]


    args['models'] = convert_to_list(args['models'])
    args['datasets'] = convert_to_list(args['datasets'])
    args['seeds'] = convert_to_list(args['seeds'], is_int=True)
    args['num_few_shots'] = convert_to_list(args['num_few_shots'], is_int=True)

    main(**args)
