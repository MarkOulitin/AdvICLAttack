from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np
from textattack.shared import AttackedText

import attack_utils as utils

class ExampleSelectionStrategy(ABC):
    @abstractmethod
    def select_example_and_update_metadata_inplace(self, icl_input: utils.ICLInput):
        pass


class FistExampleSelection(ExampleSelectionStrategy):
    def select_example_and_update_metadata_inplace(self, icl_input: utils.ICLInput):
        icl_input.attacked_text = AttackedText(icl_input.example_sentences[0])
        icl_input.pertubation_example_sentence_index = 0
        assert icl_input.attacked_text.text == icl_input.example_sentences[0]


class RandomExampleSelection(ExampleSelectionStrategy):
    rng = np.random.RandomState(0)

    def select_example_and_update_metadata_inplace(self, icl_input: utils.ICLInput):
        example_index = RandomExampleSelection.rng.choice(np.arange(0, len(icl_input.example_sentences)), 1)[0]

        icl_input.attacked_text = AttackedText(icl_input.example_sentences[example_index])
        icl_input.pertubation_example_sentence_index = example_index
        assert icl_input.attacked_text.text == icl_input.example_sentences[example_index]


class GreedyExampleSelection(ExampleSelectionStrategy):

    def __init__(self, goal_function: utils.ICLUntargetedClassification) -> None:
        super().__init__()
        self._goal_function: utils.ICLUntargetedClassification = goal_function

    def select_example_and_update_metadata_inplace(self, icl_input: utils.ICLInput):
        if len(icl_input.example_sentences) != len(icl_input.example_labels):
            raise Exception('Got sample with unequal amount of examples and labels')
        if len(icl_input.example_sentences) == 0:
            raise Exception('Got sample without examples and labels')
        if len(icl_input.example_sentences) == 1:
            raise Exception('Got sample with one example and label')

        assert self._goal_function.ground_truth_output is not None  # self._goal_function.init_attack_example method must be invoked before this method

        masked_one_examples = [
            icl_input.exclude([i]) for i in range(len(icl_input.example_sentences))
        ]

        masked_one_examples_results, search_over = self._goal_function.get_results(masked_one_examples)
        index_scores = np.array([result.score for result in masked_one_examples_results])  # scores are 1 - prob

        most_important_example_index = np.argmax(index_scores)

        icl_input.attacked_text = AttackedText(icl_input.example_sentences[most_important_example_index])
        icl_input.pertubation_example_sentence_index = most_important_example_index
        assert icl_input.attacked_text.text == icl_input.example_sentences[most_important_example_index]

# class GreedyExampleAndSentenceSelection(ExampleSelectionStrategy):
#     def __init__(self, goal_function: utils.ICLUntargetedClassification) -> None:
#         super().__init__()
#         self._goal_function: utils.ICLUntargetedClassification = goal_function
#         self.example_selection: GreedyExampleSelection = GreedyExampleSelection(goal_function)

#     def select_example(self, sample: utils.ICLInput):
#         """select example and update in place"""
#         self.example_selection.select_example_and_update_metadata_inplace(sample)
    
#     def split_text_to_sentences(self, text):
#         import nltk
#         nltk.download('punkt')  # Download the Punkt tokenizer data if not available

#         # Preprocess the text to handle emojis and punctuation marks
#         text = re.sub(r'([^\s\w.?!]|_)+', ' ', text)
        
#         # Use the NLTK tokenizer to split the text into sentences
#         sentences = nltk.sent_tokenize(text)
#         return sentences
    
#     @abstractmethod
#     def select_example_and_update_metadata_inplace(self, sample: utils.ICLInput):
#         pass
    
# class GreedyExampleAndFirstSentenceSelection(GreedyExampleAndSentenceSelection):

#     def __init__(self, goal_function: utils.ICLUntargetedClassification) -> None:
#         super().__init__(goal_function)
    
#     def select_example_and_update_metadata_inplace(self, sample: utils.ICLInput):
#         """
#         Select the most important example, then select first sentence in the example
#         """
#         self.select_example(sample)
#         sentences = self.split_text_to_sentences(sample.example_sentences[0])
#         if len(sentences) == 1:
#             # we selected the most important  example but by chance it contains only one sentence
#             return
#         selected_sentence_index = 0
#         sample.attacked_text = AttackedText(sentences[selected_sentence_index])
#         assert sample.attacked_text.text == sentences[selected_sentence_index]


# class GreedyExampleAndRandomSentenceSelection(GreedyExampleAndSentenceSelection):

#     def __init__(self, goal_function: utils.ICLUntargetedClassification) -> None:
#         super().__init__(goal_function)
    
#     def select_example_and_update_metadata_inplace(self, sample: utils.ICLInput):
#         """
#         Select the most important example, then select random sentence in the example
#         """
#         self.select_example(sample)
#         sentences = self.split_text_to_sentences(sample.example_sentences[0])
#         if len(sentences) == 1:
#             # we selected the most important  example but by chance it contains only one sentence
#             return
#         selected_sentence_index = np.random.randint(0, len(sentences))
#         sample.attacked_text = AttackedText(sentences[selected_sentence_index])
#         assert sample.attacked_text.text == sentences[selected_sentence_index]

# class GreedyExampleAndSentenceSelection(GreedyExampleAndSentenceSelection):

#     def __init__(self, goal_function: utils.ICLUntargetedClassification) -> None:
#         super().__init__(goal_function)
    
#     def select_example_and_update_metadata_inplace(self, sample: utils.ICLInput):
#         """
#         Select the most important example, then the most important sentence in the example
#         """
#         self.select_example(sample)
#         sentences = self.split_text_to_sentences(sample.example_sentences[0])
#         if len(sentences) == 1:
#             # we selected the most important  example but by chance it contains only one sentence
#             return
#         masked_one_examples = [
#             utils.ICLInput(
#                 example_sentences=[sentence],
#                 example_labels=sample.example_labels,
#                 test_sentence=sample.test_sentence,
#                 params=sample.params,
#                 pertubation_example_sentence_index = sample.pertubation_example_sentence_index,
#                 attacked_text = AttackedText(sentence),
#             )
#             for sentence in sentences
#         ]
#         masked_one_examples_results, search_over = self._goal_function.get_results(masked_one_examples)
#         index_scores = np.array([result.score for result in masked_one_examples_results])  # scores are 1 - prob
#         most_important_sentence_index = np.argmax(index_scores)

#         sample.attacked_text = AttackedText(sentences[most_important_sentence_index])
#         assert sample.attacked_text.text == sentences[most_important_sentence_index]

class ExampleImportannceOrder(ExampleSelectionStrategy):

    def __init__(self, goal_function: utils.ICLUntargetedClassification) -> None:
        super().__init__()
        self._goal_function: utils.ICLUntargetedClassification = goal_function
        self.example_selection: GreedyExampleSelection = GreedyExampleSelection(goal_function)

    def select_example_and_update_metadata_inplace(self, sample: utils.ICLInput):
        """
        Reorder examples by importance order
        """
        # TODO


class ExampleSelector:
    def __init__(self,
                 strategy: ExampleSelectionStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> ExampleSelectionStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: ExampleSelectionStrategy) -> None:
        self._strategy = strategy

    def select_example_and_update_metadata_inplace(self, icl_input: utils.ICLInput) -> None:
        self.strategy.select_example_and_update_metadata_inplace(icl_input)


def get_strategy(strategy_method: str, goal_function: utils.ICLUntargetedClassification = None) -> ExampleSelector:
    if strategy_method == 'first':
        strategy = FistExampleSelection()
    elif strategy_method == 'random':
        strategy = RandomExampleSelection()
    elif strategy_method in ['greedy-example', 'greedy-example-and-sentence']:
        if goal_function is None:
            raise Exception('to use model dependent strategy goal_function must not be None')
        strategy = _get_model_dependent_strategy(strategy_method, goal_function)
    else:
        raise Exception("got strategy that is not 'first', 'random', 'greedy-example' or 'greedy-example-and-sentence'")

    example_selector = ExampleSelector(strategy)
    return example_selector


def _get_model_dependent_strategy(strategy_method: str, goal_function: utils.ICLUntargetedClassification) -> ExampleSelectionStrategy:
    if strategy_method == 'greedy-example':
        return GreedyExampleSelection(goal_function)
    elif strategy_method == 'greedy-example-and-sentence':
        return GreedyExampleAndSentenceSelection(goal_function)
    else:
        raise Exception("got strategy that is not 'greedy-example' or 'greedy-example-and-sentence'")
