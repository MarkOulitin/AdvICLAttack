from __future__ import annotations
from llm_utils import construct_prompt, create_completion
from textattack.shared import AttackedText
from textattack.models.wrappers import ModelWrapper
from dataclasses import dataclass
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import nltk
import re
nltk.download('punkt')  # Download the Punkt tokenizer data if not available

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
            icl_example_selection_strategy_first = FistExampleSelection()
            icl_example_selector = ExampleSelector(icl_example_selection_strategy_first)
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

class ExampleSelectionStrategy(ABC):
    @abstractmethod
    def select_example_and_update_metadata_inplace(self, icl_input: ICLInput):
        pass


class FistExampleSelection(ExampleSelectionStrategy):
    def select_example_and_update_metadata_inplace(self, icl_input: ICLInput):
        icl_input.attacked_text = AttackedText(icl_input.example_sentences[0])
        icl_input.pertubation_example_sentence_index = 0
        assert icl_input.attacked_text.text == icl_input.example_sentences[0]


class RandomExampleSelection(ExampleSelectionStrategy):
    def __init__(self,
                 seed: int = 0):
        self._rng = np.random.RandomState(seed)

    def select_example_and_update_metadata_inplace(self, icl_input: ICLInput):
        example_index = self.rng.choice(np.arange(0, len(icl_input.example_sentences)), 1)[0]

        icl_input.attacked_text = AttackedText(icl_input.example_sentences[example_index])
        icl_input.pertubation_example_sentence_index = example_index
        assert icl_input.attacked_text.text == icl_input.example_sentences[example_index]



class GreedyExampleSelection(ExampleSelectionStrategy):

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
        sample.exclude(
            examples_indexes=[i for i in range(sample.example_sentences) if i != most_imporatant_example_index],
            inplace=True
        )

class GreedyExampleSentenceSelection(ExampleSelectionStrategy):

    def __init__(self, model: ICLModelWrapper) -> None:
        super().__init__()
        self._model: ICLModelWrapper = model
        self.example_selection: GreedyExampleSelection = GreedyExampleSelection(model)

    def select_example_and_update_metadata_inplace(self, sample: ICLInput):
        if len(sample.example_sentences) != len(sample.example_labels):
            raise Exception('Got sample with unequal amount of examples and labels')
        if len(sample.example_sentences) == 0:
            raise Exception('Got sample without examples and labels')
        if len(sample.example_sentences) == 1:
            raise Exception('Got sample with one example and label')
        self.example_selection.select_example_and_update_metadata_inplace(sample)
        sentences = self._split_text_to_sentences(sample.example_sentences[0])
        if len(sentences) == 1:
            # we selected the most example but by chance it contains only one sentence
            return
        min_score = self._model([sample])[0]
        most_imporatant_sentence_index = -1
        for i in range(len(sentences)):
            masked_sample: ICLInput = self._mask_sentence_of_sample(sample, sentences, i)
            score = self._model([masked_sample])[0]
            if score < min_score:
                most_imporatant_sentence_index = i
                min_score = score
        sample.example_sentences = [sentences[most_imporatant_sentence_index]]
    
    def _mask_sentence_of_sample(self, sample: ICLInput, sentences: list[str], sentence_index: int) -> ICLInput:
        return ICLInput(
            example_sentences=[' '.join([sentence for i, sentence in enumerate(sentences) if i != sentence_index])],
            example_labels=sample.example_labels, 
            test_sentence=sample.test_sentence,
            params=sample.params,
            pertubation_example_sentence_index=sample.pertubation_example_sentence_index,
            attacked_text=sample.attacked_text
        )
    
    def _split_text_to_sentences(self, text):
        # Preprocess the text to handle emojis and punctuation marks
        text = re.sub(r'([^\s\w.?!]|_)+', ' ', text)
        
        # Use the NLTK tokenizer to split the text into sentences
        sentences = nltk.sent_tokenize(text)
        return sentences

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

    def select_example_and_update_metadata_inplace(self, icl_input: ICLInput) -> None:
        self.strategy.select_example_and_update_metadata_inplace(icl_input)

def get_strategy(strategy_method: str, model: ICLModelWrapper=None) -> ExampleSelector:
    if strategy_method == 'first':
        strategy = FistExampleSelection()
    elif strategy_method == 'random':
        strategy = RandomExampleSelection()
    elif strategy_method in ['greedy-example', 'greedy-example-sentence']:
        if model is None:
            raise Exception('to use model dependent strategy model need to be not None')
        strategy = _get_model_dependent_strategy(strategy_method, model)
    else:
        raise Exception("got strategy that is not 'first', 'random', 'greedy-example' or 'greedy-example-sentence'")
    example_selector = ExampleSelector(strategy)
    return example_selector

def _get_model_dependent_strategy(strategy_method: str, model: ICLModelWrapper) -> ExampleSelectionStrategy:
    if strategy_method == 'greedy-example':
        return GreedyExampleSelection(model)
    elif strategy_method == 'greedy-example-sentence':
        return GreedyExampleSentenceSelection(model)
    else:
        raise Exception("got strategy that is not 'greedy-example' or 'greedy-example-sentence'")
