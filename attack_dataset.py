from copy import deepcopy

from textattack.datasets import Dataset

from attack_utils import ICLInput
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
