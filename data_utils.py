import pandas as pd
from utils import ROOT_DIR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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


def load_trec():
    """Loads the TREC dataset and returns the train and test sentences and labels."""

    label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}

    def _load_file(filename):
        sentences = []
        labels = []
        with open(f"{ROOT_DIR}/data/trec/{filename}", "r") as f:
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

    return train_sentences[:3000], train_labels[:3000], test_sentences, test_labels


def load_dbpedia():
    """Loads the DBpedia dataset and returns the train and test sentences and labels."""

    train_data = pd.read_csv(f"{ROOT_DIR}/data/dbpedia/train_subset.csv")
    test_data = pd.read_csv(f"{ROOT_DIR}/data/dbpedia/test.csv")

    train_sentences = train_data["Text"].tolist()
    train_sentences = [sentence.replace('""', '"') for sentence in train_sentences]
    train_labels = train_data["Class"].tolist()

    test_sentences = test_data["Text"].tolist()
    test_sentences = [sentence.replace('""', '"') for sentence in test_sentences]
    test_labels = test_data["Class"].tolist()

    train_labels = [label - 1 for label in train_labels]  # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
    test_labels = [label - 1 for label in test_labels]

    return train_sentences[:5000], train_labels[:5000], test_sentences[:1000], test_labels[:1000]


def load_sms_spam():
    """Loads the SMS-spam dataset and returns the train and test sentences and labels."""

    df = pd.read_csv(f"{ROOT_DIR}/data/sms-spam/spam.csv", encoding=('ISO-8859-1'), low_memory=False)
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

    encoder = LabelEncoder()
    encoder.fit_transform(df['target'])
    df['target'] = encoder.fit_transform(df['target'])

    X = df['text'].values
    y = df['target'].values

    train_sentences, test_sentences, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=2)

    return train_sentences.tolist(), train_labels.tolist(), test_sentences.tolist(), test_labels.tolist()


def load_parler_hate():
    df = pd.read_csv(f"{ROOT_DIR}/data/parler-hate-rate/parler_annotated_data.csv")
    df = df.drop(columns=['id', 'disputable_post'])
    df['target'] = df['label_mean'].round().astype(int)
    df = df.drop(columns=['label_mean'])

    encoder = LabelEncoder()
    encoder.fit_transform(df['target'])
    df['target'] = encoder.fit_transform(df['target'])

    X = df['text'].values
    y = df['target'].values

    train_sentences, test_sentences, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=2)
    return train_sentences.tolist(), train_labels.tolist(), test_sentences.tolist(), test_labels.tolist()


def load_ethos_binary():
    from datasets import load_dataset

    dataset = load_dataset("ethos", "binary")
    df = dataset['train'].to_pandas()

    X = df['text'].values
    y = df['label'].values

    train_sentences, test_sentences, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=2)
    return train_sentences.tolist(), train_labels.tolist(), test_sentences.tolist(), test_labels.tolist()

def load_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """

    if params['dataset'] == 'sst2':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        params['prompt_prefix'] = "Choose sentiment from positive or negative.\n\n"
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

    elif params['dataset'] == 'sms-spam':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sms_spam()
        params['prompt_prefix'] = "Choose if sms text is ham or spam.\n\n"
        params["q_prefix"] = "SMS: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['ham'], 1: ['spam']}
        params['inv_label_dict'] = {'ham': 0, 'spam': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'parler-hate':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_parler_hate()
        params['prompt_prefix'] = "Classify the posts based on their hate rate, when 1 is low hate rate and 5 is high hate rate and the possible values are 1, 2, 3, 4 and 5.\n\n"
        params["q_prefix"] = "Post: "
        params["a_prefix"] = "Hate rate: "
        params['label_dict'] = {0: ['1'], 1: ['2'], 2: ['3'], 3: ['4'], 4: ['5']}
        params['inv_label_dict'] = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'ethos':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_binary()
        params['prompt_prefix'] = "Choose if comment text's hate speech is presence or absence.\n\n"
        params["q_prefix"] = "Comment: "
        params["a_prefix"] = "Hate speech: "
        params['label_dict'] = {0: ['absence'], 1: ['presence']}
        params['inv_label_dict'] = {'absence': 0, 'presence': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    else:
        raise NotImplementedError

    return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels
