import pandas as pd
from utils import ROOT_DIR


def load_sst2():
    """Loads the SST-2 dataset and returns the train and test sentences and labels."""

    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.train", "r") as f:
        train_lines = f.readlines()
    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.test", "r") as f:
        test_lines = f.readlines()

    def _process_raw_data(lines):
        labels, sentences = [], []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return labels, sentences

    train_labels, train_sentences = _process_raw_data(train_lines)
    test_labels, test_sentences = _process_raw_data(test_lines)

    return train_sentences, train_labels, test_sentences, test_labels


def load_trec(root_dir):
    """Loads the TREC dataset and returns the train and test sentences and labels."""

    label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}

    def _load_file(filename):
        sentences = []
        labels = []
        with open(f"{root_dir}/data/trec/{filename}", "r") as f:
            for line in f:
                label = line.split(' ')[0].split(':')[0]
                label = label_dict[label]
                labels.append(label)

                sentence = ' '.join(line.split(' ')[1:]).strip()
                # basic cleaning
                sentence = sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''", '"').replace(' ?', '?').replace(' ,', ',')
                sentences.append(sentence)

        return sentences, labels

    train_sentences, train_labels = _load_file("train.txt")
    test_sentences, test_labels = _load_file("test.txt")

    return train_sentences, train_labels, test_sentences, test_labels


def load_dbpedia(root_dir):
    """Loads the DBpedia dataset and returns the train and test sentences and labels."""

    train_data = pd.read_csv(f"{root_dir}/data/dbpedia/train_subset.csv")
    test_data = pd.read_csv(f"{root_dir}/data/dbpedia/test.csv")

    train_sentences = train_data["Text"].tolist()
    train_sentences = [sentence.replace('""', '"') for sentence in train_sentences]
    train_labels = train_data["Class"].tolist()

    test_sentences = test_data["Text"].tolist()
    test_sentences = [sentence.replace('""', '"') for sentence in test_sentences]
    test_labels = test_data["Class"].tolist()

    train_labels = [label - 1 for label in train_labels]  # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
    test_labels = [label - 1 for label in test_labels]

    return train_sentences, train_labels, test_sentences, test_labels


def load_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """

    if params['dataset'] == 'sst2':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        params['prompt_prefix'] = "Choose sentiment from Positive or Negative.\n\n"
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['negative'], 1: ['positive']}
        params['inv_label_dict'] = {'negative': 0, 'positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'trec':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_trec()
        params['prompt_prefix'] = "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer Type: "
        params['label_dict'] = {0: ['Number'], 1: ['Location'], 2: ['Person'], 3: ['Description'], 4: ['Entity'], 5: ['Ab']}
        params['inv_label_dict'] = {'Number': 0, 'Location': 1, 'Person': 2, 'Description': 3, 'Entity': 4, 'Ab': 5}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'dbpedia':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_dbpedia()
        params['prompt_prefix'] = "Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['Company'], 1: ['School'], 2: ['Artist'], 3: ['Ath'], 4: ['Polit'], 5: ['Transportation'], 6: ['Building'], 7: ['Nature'], 8: ['Village'], 9: ['Animal'], 10: ['Plant'], 11: ['Album'], 12: ['Film'], 13: ['Book']}
        params['inv_label_dict'] = {'Company': 0, 'School': 1, 'Artist': 2, 'Ath': 3, 'Polit': 4, 'Transportation': 5, 'Building': 6, 'Nature': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'Book': 13}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    else:
        raise NotImplementedError


    return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels