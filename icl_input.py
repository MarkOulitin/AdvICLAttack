from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from textattack.shared import AttackedText

from icl_sample_selection import FistExampleSelection, ExampleSelector
from llm_utils import construct_prompt


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
        example_sentences = [sentence for i, sentence in enumerate(self.example_sentences) if i not in examples_indexes],
        example_labels = [label for i, label in enumerate(self.example_labels) if i not in examples_indexes],

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
