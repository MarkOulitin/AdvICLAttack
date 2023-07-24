import argparse
from copy import deepcopy

import numpy as np
import textattack
import torch

from all_examples_attack import ICLAllExamplesAttack
from attack_dataset import ICLDataset
from attack_utils import ICLModelWrapper
from data_utils import load_dataset
from icl_attaker import ICLAttacker
from llm_setups import setup_llama_hf
from llm_utils import create_completion
from utils import load_results, random_sampling


def main(models: list[str],
         datasets: list[str],
         seeds: list[int],
         num_few_shots: list[int],
         subsample_test_set: int,
         api_num_logprob: int,
         use_saved_results: bool,
         batch_size: int):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        'conditioned_on_correct_classes': True,
        'subsample_test_set': subsample_test_set,
        'api_num_logprob': api_num_logprob,
        'batch_size': batch_size
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
                    p['expr_name'] = f"all_examples_{p['dataset']}_{p['model']}_{p['num_shots']}_shot_{repr(p['subsample_test_set'])}_subsample_{p['seed']}_seed"
                    all_params.append(p)

    # query the model and save the responses
    if use_saved_results:
        load_results(all_params)
    else:
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

        attack, attack_dataset, attack_args = setup_attack_success_rate_experiment(llm,
                                                                                   device,
                                                                                   test_sentences,
                                                                                   test_labels,
                                                                                   example_sentences,
                                                                                   example_labels,
                                                                                   experiment_name,
                                                                                   params)

        attacker = ICLAttacker(experiment_name, attack, attack_dataset, attack_args)
        attacker.attack_dataset()


def setup_attack_success_rate_experiment(llm,
                                         device,
                                         test_sentences,
                                         test_labels,
                                         example_sentences,
                                         example_labels,
                                         experiment_name,
                                         params):
    icl_model_wrapper = ICLModelWrapper(llm, device)
    attack = ICLAllExamplesAttack.build(icl_model_wrapper)

    attack_dataset = ICLDataset(test_sentences, test_labels, example_sentences, example_labels,
                                params['num_shots'], params)

    attack_args = textattack.AttackArgs(
        num_examples=len(attack_dataset),
        log_to_txt=f"./log_{experiment_name}.txt",
        #log_to_csv=f"./log_{experiment_name}.csv",
        log_summary_to_json=f"./attack_summary_log_{experiment_name}.json",
        csv_coloring_style="plain",
        checkpoint_interval=100,
        checkpoint_dir="./checkpoints",
        disable_stdout=True,
        random_seed=params['seed'],
        # query_budget=60,  # TODO remove
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

            if model_name == 'llama_cpp':
                response = create_completion(prompt, params['inv_label_dict'], 1,
                                             model_name, device, llm, echo=True, num_logprobs=1)  # set echo to True
                first_token_of_label_name = response['choices'][0]['logprobs']['tokens'][0][1:]  # without the prefix space

                if first_token_of_label_name != label_name:
                    print('label name is more than 1 token', label_name)
                    assert False

            elif model_name == 'llama_hf':
                label_to_token = llm.label_to_token
                assert label_name in label_to_token
                # response, probs = create_completion(prompt, params['inv_label_dict'], 1,
                #                                     model_name, device, echo=True, num_logprobs=1,
                #                                     llm=llm)  # set echo to True
                # first_token_of_label_name = response[2]  # take the third token, first tokens are "<s> "

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
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_logprob', dest='api_num_logprob', action='store', required=False, type=int,
                        default=100,
                        help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--batch_size', dest='batch_size', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_const', const=True, default=False,
                        help='whether to load the results from pickle files and not run the model')

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
