from copy import deepcopy

import numpy as np
from textattack.attack_recipes import TextBuggerLi2018
from textattack.attack_results import SkippedAttackResult
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared import AttackedText
from textattack.shared.validators import transformation_consists_of_word_swaps_and_deletions
from textattack.transformations import CompositeTransformation, WordSwapRandomCharacterInsertion, \
    WordSwapRandomCharacterDeletion, WordSwapNeighboringCharacterSwap, WordSwapHomoglyphSwap, WordSwapEmbedding

from attack_utils import ICLUntargetedClassification, ICLClassificationGoalFunctionResult
from icl_input import ICLInput
from icl_sample_selection import get_strategy


class ICLAllExamplesAttack(TextBuggerLi2018):
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
        search_method = ICLAllExamplesSearchMethod()

        return ICLAllExamplesAttack(goal_function, constraints, transformation, search_method)

    def attack(self, icl_input: ICLInput, ground_truth_output):
        assert isinstance(
            ground_truth_output, (int, str)
        ), "`ground_truth_output` must either be `str` or `int`."

        goal_function_result, _ = self.goal_function.init_attack_example(
            icl_input, ground_truth_output
        )

        if goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
            return SkippedAttackResult(goal_function_result)
        else:
            # default strategy, choose first icl example for the attack
            icl_example_selector = get_strategy('first')
            icl_example_selector.select_example_and_update_metadata_inplace(goal_function_result.icl_input)

            result = self._attack(goal_function_result)
            return result


class ICLAllExamplesSearchMethod(SearchMethod):
    def __init__(self, example_perturbation_bounds: int = 3):
        self.example_perturbation_bounds = example_perturbation_bounds

        super().__init__()

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

    def perform_search(self, initial_result: ICLClassificationGoalFunctionResult):
        def update_attacked_text(result, initial_result, example_index):
            result.icl_input.pertubation_example_sentence_index = example_index
            result.icl_input.attacked_text = AttackedText(result.icl_input.example_sentences[example_index])
            result.attacked_text = result.icl_input.attacked_text

            assert result.icl_input.attacked_text.text == result.icl_input.example_sentences[example_index]
            assert result.attacked_text.text == result.icl_input.example_sentences[example_index]

            initial_result.icl_input.pertubation_example_sentence_index = example_index
            initial_result.icl_input.attacked_text = AttackedText(initial_result.icl_input.example_sentences[example_index])
            initial_result.attacked_text = initial_result.icl_input.attacked_text

        def override_example_sentence(result):
            result.icl_input.example_sentences[result.icl_input.pertubation_example_sentence_index] = result.icl_input.attacked_text.text

        initial_result_copy = deepcopy(initial_result)
        icl_input = initial_result_copy.icl_input

        cur_result = initial_result_copy

        for example_index in range(len(icl_input.example_sentences)):
            # update attacked text
            update_attacked_text(cur_result, initial_result, example_index)

            # Sort words by order of importance of given example index
            index_order, search_over = self._get_index_order(deepcopy(cur_result.icl_input))
            i = 0
            improve_score_pertubation_num = 0

            index_order = index_order
            while i < len(index_order) and improve_score_pertubation_num < self.example_perturbation_bounds and not search_over:
                transformed_text_candidates = self.get_transformations(
                    cur_result.attacked_text,
                    original_text=initial_result.attacked_text,
                    indices_to_modify=[index_order[i]],
                )

                i += 1
                if len(transformed_text_candidates) == 0:
                    continue
                transformed_icl_candidates = [ICLInput(cur_result.icl_input.example_sentences,
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
                    # update pertubation num up until example_perturbation_bounds
                    improve_score_pertubation_num += 1

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
