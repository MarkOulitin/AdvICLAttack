import argparse
from copy import deepcopy

import numpy as np
import textattack
import torch

from attack_dataset import ICLDataset
from attack_utils import ICLModelWrapper, ICLAttack
from data_utils import load_dataset
from llm_setups import setup_llama
from llm_utils import create_completion, get_model_response
from utils import load_results, random_sampling, save_experiment_data_pickle


def main(models: list[str],
         datasets: list[str],
         num_few_shots: list[int],
         seeds: list[int],
         subsample_test_set: int,
         api_num_logprob: int,
         approx: bool,
         use_saved_results: bool,
         batch_size: int):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        'conditioned_on_correct_classes': True,
        'subsample_test_set': subsample_test_set,
        'api_num_logprob': api_num_logprob,
        'approx': approx,
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
                    p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                    all_params.append(p)

    # query the model and save the responses
    if use_saved_results:
        load_results(all_params)
    else:
        run_experiments(all_params)


def run_experiments(params_list):
    """
    Run the experiments and save its responses and the rest of configs into a pickle file
    """

    # result_tree = dict()
    for param_index, params in enumerate(params_list):
        print("\nExperiment name:", params['expr_name'])

        # sanity check
        experiment_sanity_check(params)

        # load data
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset(params)

        # sample test set
        if params['subsample_test_set'] is None:
            print(f"selecting full test set ({len(all_test_labels)} examples)")
            test_sentences, test_labels = all_test_sentences, all_test_labels
        else:
            print(f"selecting {len(test_labels)} subsample of test set")

            test_sentences, test_labels = random_sampling(all_test_sentences,
                                                          all_test_labels,
                                                          params['subsample_test_set'])

        # set seed
        np.random.seed(params['seed'])

        device = "cpu" if not torch.cuda.is_available() else "cuda"
        llm = setup_llama(device)

        icl_model_wrapper = ICLModelWrapper(llm, device)
        attack = ICLAttack.build(icl_model_wrapper)

        attack_dataset = ICLDataset(test_sentences, test_labels, all_train_sentences, all_train_labels,
                                    params['num_shots'], params)

        attack_args = textattack.AttackArgs(
            log_to_csv="log.csv",
            checkpoint_interval=5,
            checkpoint_dir="checkpoints",
            disable_stdout=True
        )
        attacker = textattack.Attacker(attack, attack_dataset, attack_args)
        attacker.attack_dataset()

        #for test_sentence in test_sentences:
            # # sample few-shot training examples
            # train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'])

            # choose important example based on their importance score

            # sort important words based on their importance score

            # loop until success - generate bug then evaluate

            # obtaining model's response
            #raw_resp_test = get_model_response(params, train_sentences, train_labels, list(test_sentence))

            # get prob for each label
            # TODO
            #all_label_probs = get_label_probs(params, raw_resp_test, train_sentences, train_labels, test_sentences)

        # calculate metrics
        # TODO - acc = eval_accuracy(all_label_probs, test_labels)

        # # add to result_tree
        # keys = [params['dataset'], params['model'], params['num_shots']]
        # node = result_tree  # root
        # for k in keys:
        #     if not (k in node.keys()):
        #         node[k] = dict()
        #     node = node[k]
        # node[params['seed']] = acc
        #
        # # save to file
        # result_to_save = dict()
        # params_to_save = deepcopy(params)
        # result_to_save['params'] = params_to_save
        # result_to_save['train_sentences'] = train_sentences
        # result_to_save['train_labels'] = train_labels
        # result_to_save['test_sentences'] = test_sentences
        # result_to_save['test_labels'] = test_labels
        # result_to_save['raw_resp_test'] = raw_resp_test
        # result_to_save['all_label_probs'] = all_label_probs
        # result_to_save['acc'] = acc
        # if 'prompt_func' in result_to_save['params'].keys():
        #     params_to_save['prompt_func'] = None
        # save_experiment_data_pickle(params, result_to_save)


# def eval_accuracy(all_label_probs, test_labels):
#     raise NotImplementedError
#

def experiment_sanity_check(params):
    """Sanity check the experiment"""

    assert params['num_tokens_to_predict'] == 1

    # for classification, make sure that all of the class names are one word.
    for key, label_names in params['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            prompt = label_name
            response = create_completion(prompt, 1, params['model'], echo=True, num_log_probs=1)  # set echo to True
            first_token_of_label_name = response['choices'][0]['logprobs']['tokens'][0][1:]  # without the prefix space
            if first_token_of_label_name != label_name:
                print('label name is more than 1 token', label_name)
                assert False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True,
                        help='name of model(s), e.g., Llama')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True,
                        help='name of dataset(s), e.g., sst2')
    parser.add_argument('--seeds', dest='seeds', action='store', required=True,
                        help='seeds for the training set', type=int)
    parser.add_argument('--num_few_shots', dest='all_shots', action='store', required=True,
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
