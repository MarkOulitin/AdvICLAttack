from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import textattack
import torch
from textattack.attack_recipes import TextBuggerLi2018
from textattack.attack_results import SkippedAttackResult
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_function_results import GoalFunctionResultStatus, ClassificationGoalFunctionResult
from textattack.goal_functions import UntargetedClassification
from textattack.models.wrappers import ModelWrapper
from textattack.search_methods import SearchMethod
from textattack.shared import AttackedText
from textattack.shared.validators import transformation_consists_of_word_swaps_and_deletions
from textattack.transformations import CompositeTransformation, WordSwapRandomCharacterInsertion, \
    WordSwapRandomCharacterDeletion, WordSwapNeighboringCharacterSwap, WordSwapHomoglyphSwap, WordSwapEmbedding

from llm_utils import construct_prompt, create_completion
from abc import ABC, abstractmethod


@dataclass
class ICLInput:
    example_sentences: list[str]
    example_labels: list[int]
    test_sentence: str
    params: dict
    pertubation_example_sentence_index: int = -1
    attacked_text: AttackedText = None

    def __post_init__(self):
        if self.attacked_text is None:
            # default init of the first example
            icl_example_selection_strategy_first = ICLExampleSelectionStrategyFirst()
            icl_example_selector = ICLExampleSelector(icl_example_selection_strategy_first)
            icl_example_selector.select_example_and_update_metadata_inplace(self)

    def exclude(self, examples_indexes: list[int], inplace: bool = False) -> ICLInput:
        example_sentences=[sentence for i, sentence in enumerate(self.example_sentences) if i not in examples_indexes],
        example_labels=[label for i, label in enumerate(self.example_labels) if i not in examples_indexes],
        if not inplace:
            return ICLInput(
                example_sentences=example_sentences,
                example_labels=example_labels,
                test_sentence=self.test_sentence,
                params=self.params,
                pertubation_example_sentence_index=self.pertubation_example_sentence_index,
                attacked_text=self.attacked_text
            )
        else:
            self.example_sentences = example_sentences
            self.example_labels = example_labels
            return self


    def construct_prompt(self) -> str:
        assert self.attacked_text is not None
        assert self.pertubation_example_sentence_index != -1

        example_sentences_with_pertubation = deepcopy(self.example_sentences)
        pertubation_sentence_index = self.pertubation_example_sentence_index
        pertubation_sentence = self.attacked_text.text
        example_sentences_with_pertubation[pertubation_sentence_index] = pertubation_sentence

        prompt = construct_prompt(self.params,
                                  example_sentences_with_pertubation,
                                  self.example_labels,
                                  self.test_sentence)

        return prompt


class ICLExampleSelectionStrategy(ABC):
    @abstractmethod
    def select_example_and_update_metadata_inplace(self, icl_input: ICLInput):
        pass


class ICLExampleSelectionStrategyFirst(ICLExampleSelectionStrategy):
    def select_example_and_update_metadata_inplace(self, icl_input: ICLInput):
        icl_input.attacked_text = AttackedText(icl_input.example_sentences[0])
        icl_input.pertubation_example_sentence_index = 0
        assert icl_input.attacked_text.text == icl_input.example_sentences[0]


class ICLExampleSelectionStrategyRandom(ICLExampleSelectionStrategy):
    def __init__(self,
                 seed: int = 0):
        self._rng = np.random.RandomState(seed)

    def select_example_and_update_metadata_inplace(self, icl_input: ICLInput):
        example_index = self.rng.choice(np.arange(0, len(icl_input.example_sentences)), 1)[0]

        icl_input.attacked_text = AttackedText(icl_input.example_sentences[example_index])
        icl_input.pertubation_example_sentence_index = example_index
        assert icl_input.attacked_text.text == icl_input.example_sentences[example_index]


class ICLExampleSelectionStrategyGreedy(ICLExampleSelectionStrategy):

    def __init__(self, model: ICLModelWrapper) -> None:
        super().__init__()
        self._model: ICLModelWrapper = model

    def select_example_and_update_metadata_inplace(self, sample: ICLInput):
        if len(sample.example_sentences) != len(sample.example_labels):
            raise Exception('Got sample with unequal amount of examples and labels')
        if len(sample.example_sentences) == 0:
            raise Exception('Got sample without examples and labels')
        if len(sample.example_sentences) == 1:
            raise Exception('Got sample with one example and label')
        min_score = self._model([sample])[0]
        most_imporatant_example_index = -1
        for i in range(len(sample.example_sentences)):
            masked_sample: ICLInput = sample.exclude([i])
            score = self._model([masked_sample])[0]
            if score < min_score:
                most_imporatant_example_index = i
                min_score = score
        return sample.exclude(
            examples_indexes=[i for i in range(sample.example_sentences) if i != most_imporatant_example_index],
            inplace=True
        )


class ICLExampleSelector:
    def __init__(self,
                 strategy: ICLExampleSelectionStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> ICLExampleSelectionStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: ICLExampleSelectionStrategy) -> None:
        self._strategy = strategy

    def select_example_and_update_metadata_inplace(self, icl_input: ICLInput) -> None:
        self.strategy.select_example_and_update_metadata_inplace(icl_input)


class ICLClassificationGoalFunctionResult(ClassificationGoalFunctionResult):
    """Represents the result of a classification goal function."""

    def __init__(
            self,
            icl_input: ICLInput,
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


class ICLAttack(TextBuggerLi2018):
    @staticmethod
    def build(model_wrapper):
        #
        #  we propose five bug generation methods for TEXTBUGGER:
        #
        transformation = CompositeTransformation(
            [
                # (1) Insert: Insert a space into the word.
                # Generally, words are segmented by spaces in English. Therefore,
                # we can deceive classifiers by inserting spaces into words.
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # (2) Delete: Delete a random character of the word except for the first
                # and the last character.
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (3) Swap: Swap random two adjacent letters in the word but do not
                # alter the first or last letter. This is a common occurrence when
                # typing quickly and is easy to implement.
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (4) Substitute-C (Sub-C): Replace characters with visually similar
                # characters (e.g., replacing “o” with “0”, “l” with “1”, “a” with “@”)
                # or adjacent characters in the keyboard (e.g., replacing “m” with “n”).
                WordSwapHomoglyphSwap(),
                # (5) Substitute-W
                # (Sub-W): Replace a word with its topk nearest neighbors in a
                # context-aware word vector space. Specifically, we use the pre-trained
                # GloVe model [30] provided by Stanford for word embedding and set
                # topk = 5 in the experiment.
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        constraints = [RepeatModification(), StopwordModification()]
        # In our experiment, we first use the Universal Sentence
        # Encoder [7], a model trained on a number of natural language
        # prediction tasks that require modeling the meaning of word
        # sequences, to encode sentences into high dimensional vectors.
        # Then, we use the cosine similarity to measure the semantic
        # similarity between original texts and adversarial texts.
        # ... "Furthermore, the semantic similarity threshold \eps is set
        # as 0.8 to guarantee a good trade-off between quality and
        # strength of the generated adversarial text."
        constraints.append(UniversalSentenceEncoder(threshold=0.8))
        #
        # Goal is ICL untargeted classification
        #
        goal_function = ICLUntargetedClassification(model_wrapper, use_cache=False)
        #
        # ICL Greedily swap words with "Word Importance Ranking".
        #
        search_method = ICLGreedyWordSwapWIR()

        return ICLAttack(goal_function, constraints, transformation, search_method)

    def attack(self, icl_input: ICLInput, ground_truth_output, example_selection_strategy: str = None):
        assert isinstance(
            ground_truth_output, (int, str)
        ), "`ground_truth_output` must either be `str` or `int`."

        goal_function_result, _ = self.goal_function.init_attack_example(
            icl_input, ground_truth_output
        )
        if goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
            return SkippedAttackResult(goal_function_result)
        else:
            # default strategy, choose random icl example for the attack
            if example_selection_strategy is None:
                # TODO change to random and not to first
                icl_example_selection_strategy_first = ICLExampleSelectionStrategyFirst()
                icl_example_selector = ICLExampleSelector(icl_example_selection_strategy_first)
                icl_example_selector.select_example_and_update_metadata_inplace(icl_input)
            else:
                strategies = {
                    'first': ICLExampleSelectionStrategyFirst(),
                    'random': ICLExampleSelectionStrategyRandom(),
                    'greedy': ICLExampleSelectionStrategyGreedy(self.goal_function.model),
                }
                if example_selection_strategy not in list(strategies.keys()):
                    raise Exception("got strategy that is not 'first', 'random' or 'greedy'")
                icl_example_selection_strategy = strategies[example_selection_strategy]
                icl_example_selector = ICLExampleSelector(icl_example_selection_strategy)
                icl_example_selector.select_example_and_update_metadata_inplace(icl_input)

            result = self._attack(goal_function_result)
            return result


class ICLUntargetedClassification(UntargetedClassification):
    def init_attack_example(self, icl_input, ground_truth_output):
        """Called before attacking ``attacked_text`` to 'reset' the goal
        function and set properties for this example."""
        self.initial_attacked_text = icl_input
        self.ground_truth_output = ground_truth_output
        self.num_queries = 0
        result, _ = self.get_result(icl_input, check_skip=True)
        return result, _

    def get_results(self, icl_input_list: list[ICLInput], check_skip=False):
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
            goal_function_score = self._get_score(raw_output, icl_input) # TODO check 1-score
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

    def _call_model_uncached(self, attacked_text_list: list[ICLInput]) -> np.ndarray:
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
    def _get_index_order(self, initial_icl_input: ICLInput):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""

        initial_text = initial_icl_input.attacked_text
        len_text, indices_to_order = self.get_indices_to_order(initial_text)

        leave_one_texts = [
            initial_text.delete_word_at_index(i) for i in indices_to_order
        ]
        leave_one_icl_inputs = [ICLInput(initial_icl_input.example_sentences,
                                         initial_icl_input.example_labels,
                                         initial_icl_input.test_sentence,
                                         initial_icl_input.params,
                                         initial_icl_input.pertubation_example_sentence_index,
                                         leave_one_text) for leave_one_text in leave_one_texts]

        leave_one_results, search_over = self.get_goal_results(leave_one_icl_inputs)
        index_scores = np.array([result.score for result in leave_one_results])

        index_order = np.array(indices_to_order)[(-index_scores).argsort()]

        return index_order, search_over

    def perform_search(self, initial_result: ICLClassificationGoalFunctionResult): # TODO check skip for cases of incorrect intiial labeling
        icl_input = initial_result.icl_input

        # Sort words by order of importance
        index_order, search_over = self._get_index_order(icl_input)
        i = 0
        cur_result = deepcopy(initial_result)

        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue

            transformed_icl_candidates = [ICLInput(icl_input.example_sentences,
                                                   icl_input.example_labels,
                                                   icl_input.test_sentence,
                                                   icl_input.params,
                                                   icl_input.pertubation_example_sentence_index,
                                                   transformed_text_candidate) for transformed_text_candidate in
                                          transformed_text_candidates]

            results, search_over = self.get_goal_results(transformed_icl_candidates)
            results = sorted(results, key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
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
                return best_result

        return cur_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        return True


class ICLModelWrapper(ModelWrapper):
    def __init__(self, llm_model, device):
        self.model = llm_model
        self.device = device

    def __call__(self, icl_input_list: list[ICLInput]):
        outputs_probs = []

        for icl_input in icl_input_list:
            params = icl_input.params
            # num_classes = len(params['label_dict'].keys())

            prompt = icl_input.construct_prompt()
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
