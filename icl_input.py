from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from textattack.shared import AttackedText

import icl_sample_selection as example_selector

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
            icl_example_selection_strategy_first = example_selector.FistExampleSelection()
            icl_example_selector = example_selector.ExampleSelector(icl_example_selection_strategy_first)
            icl_example_selector.select_example_and_update_metadata_inplace(self)

    def exclude(self, example_index: int) -> ICLInput:
        result = ICLInput(
            example_sentences=[self.example_sentences[example_index]],
            example_labels=[self.example_labels[example_index]],
            test_sentence=self.test_sentence,
            params=self.params,
            pertubation_example_sentence_index = example_index,
            attacked_text = AttackedText(self.example_sentences[example_index]),
        )
        assert result.attacked_text.text == self.example_sentences[example_index]
        return result

    def construct_prompt(self, ignore_attacked_text: bool = False) -> str:
        assert self.attacked_text is not None
        assert self.pertubation_example_sentence_index != -1

        example_sentences_with_pertubation = deepcopy(self.example_sentences)
        pertubation_sentence_index = self.pertubation_example_sentence_index
        pertubation_sentence = self.attacked_text.text
        if not ignore_attacked_text:
            example_sentences_with_pertubation[pertubation_sentence_index] = pertubation_sentence

        prompt = construct_prompt(self.params,
                                  example_sentences_with_pertubation,
                                  self.example_labels,
                                  self.test_sentence)

        return prompt
