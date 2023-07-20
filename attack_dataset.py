import random
from copy import deepcopy

from textattack.datasets import Dataset
from textattack.shared import AttackedText

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
            original_demonstration: str,
            adversarial_demonstration: str,
            adversarial_demonstration_index: int,
            test_sentences: list[str],
            test_labels: list[int],
            all_train_sentences: list[str],
            all_train_labels: list[int],
            num_shots: int,
            params: dict
    ):
        self._original_demonstration = original_demonstration
        self._adversarial_demonstration = adversarial_demonstration
        self._adversarial_demonstration_index = adversarial_demonstration_index
        self._test_sentences = deepcopy(test_sentences)
        self._test_labels = test_labels
        self._all_example_sentences = deepcopy(all_train_sentences)
        self._all_example_labels = all_train_labels
        self._num_shots = num_shots
        self._params = deepcopy(params)
        self.shuffled = False

        self._generate_examples_using_adversarial_demonstration()

    def _generate_examples_using_adversarial_demonstration(self, shuffle_example_seed: int = 0):
        # sample few-shot training examples without given demonstration
        example_sentences, example_labels = random_sampling(self._all_example_sentences,
                                                            self._all_example_labels,
                                                            self._num_shots - 1,
                                                            exclude_index=self._adversarial_demonstration_index)
        example_sentences.append(self._original_demonstration)
        demonstration_label = self._all_example_labels[self._adversarial_demonstration_index]
        example_labels.append(demonstration_label)

        # Set the random seed
        random.seed(shuffle_example_seed)
        # Shuffle the examples list
        random.shuffle(example_sentences)
        # Apply the new order to the labels list
        example_labels = [example_labels[i] for i in range(len(example_labels))]

        self._example_sentences = example_sentences
        self._example_labels = example_labels

    def __getitem__(self, i):
        assert len(self._example_sentences)
        assert len(self._example_labels)

        test_sentence = deepcopy(self._test_sentences[i])
        test_label = self._test_labels[i]

        pertubation_sentence_index = self._example_sentences.index(self._original_demonstration)
        attacked_text = AttackedText(self._adversarial_demonstration)

        icl_input = ICLInput(self._example_sentences,
                             self._example_labels,
                             test_sentence,
                             deepcopy(self._params),
                             pertubation_sentence_index,
                             attacked_text)

        return icl_input, test_label

    def __len__(self):
        """Returns the size of dataset."""
        return len(self._test_labels)