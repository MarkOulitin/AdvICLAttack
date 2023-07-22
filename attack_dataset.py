from copy import deepcopy

from textattack.datasets import Dataset

from icl_input import ICLInput
from utils import random_sampling


class ICLDataset(Dataset):
    def __init__(
            self,
            test_sentences: list[str],
            test_labels: list[int],
            all_train_sentences: list[str],
            all_train_labels: list[int],
            num_shots: int,
            params: dict
    ):
        self._test_sentences = deepcopy(test_sentences)
        self._test_labels = test_labels
        self._all_train_sentences = deepcopy(all_train_sentences)
        self._all_train_labels = all_train_labels
        self._num_shots = num_shots
        self._params = deepcopy(params)
        self.shuffled = False

    def __getitem__(self, i):
        test_sentence = deepcopy(self._test_sentences[i])
        test_label = self._test_labels[i]

        # sample few-shot training examples
        train_sentences, train_labels = random_sampling(self._all_train_sentences,
                                                        self._all_train_labels,
                                                        self._num_shots)

        icl_input = ICLInput(train_sentences, train_labels, test_sentence, deepcopy(self._params))

        return icl_input, test_label

    def __len__(self):
        """Returns the size of dataset."""
        return len(self._test_labels)


class ICLTransferabilityDataset(Dataset):
    def __init__(
            self,
            original_examples: list[str],
            adversarial_examples: list[str],
            adversarial_example_indices: list[int],
            test_sentences: list[str],
            test_labels: list[int],
            all_train_sentences: list[str],
            all_train_labels: list[int],
            num_shots: int,
            params: dict
    ):
        self._original_examples = original_examples
        self._adversarial_examples = adversarial_examples
        self._adversarial_example_indices = adversarial_example_indices
        self._test_sentences = deepcopy(test_sentences)
        self._test_labels = test_labels
        self._all_example_sentences = deepcopy(all_train_sentences)
        self._all_example_labels = all_train_labels
        self._num_shots = num_shots
        self._params = deepcopy(params)
        self.shuffled = False

        print(f"original examples: {self._original_examples}")
        print(f"adversarial examples: {self._adversarial_examples}")
        print(f"adversarial example index: {self._adversarial_example_indices}")

        self._examples_labels = [self._all_example_labels[adversarial_example_index] for adversarial_example_index in self._adversarial_example_indices]

    def __getitem__(self, i):
        assert len(self._adversarial_examples)
        assert len(self._examples_labels)

        test_sentence = self._test_sentences[i]
        test_label = self._test_labels[i]

        attacked_icl_input = ICLInput(deepcopy(self._adversarial_examples),
                                      deepcopy(self._examples_labels),
                                      deepcopy(test_sentence),
                                      deepcopy(self._params))

        original_icl_input = ICLInput(deepcopy(self._original_examples),
                                      deepcopy(self._examples_labels),
                                      deepcopy(test_sentence),
                                      deepcopy(self._params))

        return attacked_icl_input, original_icl_input, test_label

    def __len__(self):
        """Returns the size of dataset."""
        return len(self._test_labels)