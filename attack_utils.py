from __future__ import annotations

from copy import deepcopy

import numpy as np
import textattack
import torch
from textattack.goal_function_results import GoalFunctionResultStatus, ClassificationGoalFunctionResult
from textattack.goal_functions import UntargetedClassification
from textattack.models.wrappers import ModelWrapper
from textattack.search_methods import SearchMethod
from textattack.shared.validators import transformation_consists_of_word_swaps_and_deletions

import icl_input as attack_input
from llm_utils import create_completion


class ICLClassificationGoalFunctionResult(ClassificationGoalFunctionResult):
    """Represents the result of a classification goal function."""

    def __init__(
            self,
            icl_input: attack_input.ICLInput,
            raw_output,
            output,
            goal_status,
            score,
            num_queries,
            ground_truth_output,
    ):
        self.icl_input = icl_input

        super().__init__(
            icl_input.attacked_text,
            raw_output,
            output,
            goal_status,
            score,
            num_queries,
            ground_truth_output
        )

    @property
    def _processed_output(self):
        """Takes a model output (like `1`) and returns the class labeled output
        (like `positive`), if possible.

        Also returns the associated color.
        """
        output_label = self.raw_output.argmax()
        output = self.icl_input.params["label_dict"][self.output][0]
        output = textattack.shared.utils.process_label_name(output)
        color = textattack.shared.utils.color_from_output(output, output_label)
        return output, color


class ICLUntargetedClassification(UntargetedClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_attack_example(self, icl_input, ground_truth_output):
        """Called before attacking ``attacked_text`` to 'reset' the goal
        function and set properties for this example."""
        self.initial_attacked_text = icl_input
        self.ground_truth_output = ground_truth_output
        self.num_queries = 0
        result, _ = self.get_result(icl_input, check_skip=True)
        return result, _

    def get_results(self, icl_input_list: list[attack_input.ICLInput], check_skip=False):
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries
            icl_input_list = icl_input_list[:queries_left]
        self.num_queries += len(icl_input_list)
        model_outputs = self._call_model(icl_input_list)
        for icl_input, raw_output in zip(icl_input_list, model_outputs):
            displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, icl_input, check_skip=check_skip
            )
            goal_function_score = self._get_score(raw_output, icl_input)
            results.append(
                self._goal_function_result_type()(
                    icl_input,
                    raw_output,
                    displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output,
                )
            )
        return results, self.num_queries == self.query_budget

    def _call_model_uncached(self, attacked_text_list: list[attack_input.ICLInput]) -> np.ndarray:
        """Queries model and returns outputs for a list of AttackedText
        objects."""
        if not len(attacked_text_list):
            return []

        # inputs = [at.tokenizer_input for at in attacked_text_list]
        inputs = attacked_text_list
        outputs = []
        i = 0
        while i < len(inputs):
            batch = inputs[i : i + self.batch_size]
            batch_preds = self.model(batch)

            # Some seq-to-seq models will return a single string as a prediction
            # for a single-string list. Wrap these in a list.
            if isinstance(batch_preds, str):
                batch_preds = [batch_preds]

            # Get PyTorch tensors off of other devices.
            if isinstance(batch_preds, torch.Tensor):
                batch_preds = batch_preds.cpu()

            if isinstance(batch_preds, list):
                outputs.extend(batch_preds)
            elif isinstance(batch_preds, np.ndarray):
                outputs.append(torch.tensor(batch_preds))
            else:
                outputs.append(batch_preds)
            i += self.batch_size

        if isinstance(outputs[0], torch.Tensor):
            outputs = torch.cat(outputs, dim=0)

        assert len(inputs) == len(
            outputs
        ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

        return self._process_model_outputs(attacked_text_list, outputs)

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return ICLClassificationGoalFunctionResult


class ICLGreedyWordSwapWIR(SearchMethod):
    def _get_index_order(self, initial_icl_input: attack_input.ICLInput):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""

        initial_text = initial_icl_input.attacked_text
        len_text, indices_to_order = self.get_indices_to_order(initial_text)

        leave_one_texts = [
            initial_text.delete_word_at_index(i) for i in indices_to_order
        ]
        leave_one_icl_inputs = [attack_input.ICLInput(initial_icl_input.example_sentences,
                                                      initial_icl_input.example_labels,
                                                      initial_icl_input.test_sentence,
                                                      initial_icl_input.params,
                                                      initial_icl_input.pertubation_example_sentence_index,
                                                      leave_one_text) for leave_one_text in leave_one_texts]

        leave_one_results, search_over = self.get_goal_results(leave_one_icl_inputs)
        index_scores = np.array([result.score for result in leave_one_results])

        index_order = np.array(indices_to_order)[(-index_scores).argsort()]

        return index_order, search_over

    def perform_search(self, initial_result: ICLClassificationGoalFunctionResult):
        def override_example_sentence(result):
            result.icl_input.example_sentences[result.icl_input.pertubation_example_sentence_index] = result.attacked_text.text

        initial_result_copy = deepcopy(initial_result)
        icl_input = initial_result_copy.icl_input

        # Sort words by order of importance
        index_order, search_over = self._get_index_order(icl_input)
        i = 0
        cur_result = initial_result_copy

        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue

            transformed_icl_candidates = [attack_input.ICLInput(cur_result.icl_input.example_sentences,
                                                                cur_result.icl_input.example_labels,
                                                                cur_result.icl_input.test_sentence,
                                                                cur_result.icl_input.params,
                                                                cur_result.icl_input.pertubation_example_sentence_index,
                                                                transformed_text_candidate) for transformed_text_candidate in
                                          transformed_text_candidates]

            results, search_over = self.get_goal_results(transformed_icl_candidates)
            results = sorted(results, key=lambda x: -x.score)

            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
                # override example sentence
                override_example_sentence(cur_result)
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.icl_input.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        # If the attack was run without any similarity metrics,
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result

                # override example sentence
                override_example_sentence(cur_result)
                return best_result

        # override example sentence
        override_example_sentence(cur_result)
        return cur_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        return True


class ICLModelWrapper(ModelWrapper):
    def __init__(self, llm_model, device, ignore_attacked_text: bool = False):
        self.model = llm_model
        self.device = device
        self.ignore_attacked_text = ignore_attacked_text

    def __call__(self, icl_input_list: list[attack_input.ICLInput]):
        outputs_probs = []

        for icl_input in icl_input_list:
            params = icl_input.params

            prompt = icl_input.construct_prompt(self.ignore_attacked_text)
            llm_response, label_probs = create_completion(prompt,
                                                          params['inv_label_dict'],
                                                          params['num_tokens_to_predict'],
                                                          params['model'],
                                                          self.device,
                                                          self.model,
                                                          num_logprobs=params['api_num_logprob'])
            # llm_response = llm_response['choices'][0]  # llama cpp logic
            # label_probs = get_probs(params, num_classes, llm_response)
            outputs_probs.append(label_probs)

        outputs_probs = np.array(outputs_probs)  # probs are normalized

        return outputs_probs
