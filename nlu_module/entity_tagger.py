import re
import sklearn_crfsuite

from util.logger import print_log

def ent2bio(data: list, verbose: bool = False):
    bio_data = []
    for sample in data:
        if verbose:
            print(sample)
        matches = re.findall(r"\[(.*?)\]", sample)
        entity = [match for match in matches]

        matches = re.findall(r"\((.*?)\)", sample)
        label = [match for match in matches]

        if len(entity) == len(label):
            tmp_sample = re.sub(r"\[(.*?)\]", "XX", sample)
            tmp_sample = re.sub(r"\((.*?)\)", "", tmp_sample)
            token = []
            bio_token = []
            ent_idx = 0
            lab_idx = 0
            for tok in tmp_sample.split():
                if tok not in ["XX", "OO"]:
                    token.append(tok)
                    bio_token.append("O")

                elif tok == "XX":
                    ent_tok = entity[ent_idx].split()
                    token.extend(ent_tok)
                    bio_token = (
                        bio_token
                        + [f"B-{label[lab_idx].upper()}"]
                        + [f"I-{label[lab_idx].upper()}"] * int(len(ent_tok) - 1)
                    )

                    ent_idx += 1
                    lab_idx += 1

            bio_data.append((token, bio_token))

    return bio_data


def bio2ent(tokens, tags):
    entities = []

    current = []
    current_type = None

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current:
                entities.append(
                    (current_type, " ".join(current))
                )

            current_type = tag[2:]
            current = [token]

        elif tag.startswith("I-"):
            current.append(token)

        else:
            if current:
                entities.append(
                    (current_type, " ".join(current))
                )

            current = []
            current_type = None

    if current:
        entities.append(
            (current_type, " ".join(current))
        )

    return entities


def word2features(sent, i):
    word = sent[i]

    features = {
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }

    if i > 0:
        prev = sent[i - 1]
        features.update({
            "-1:word.lower()": prev.lower(),
            "-1:word.istitle()": prev.istitle(),
        })
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        nxt = sent[i + 1]
        features.update({
            "+1:word.lower()": nxt.lower(),
            "+1:word.istitle()": nxt.istitle(),
        })
    else:
        features["EOS"] = True

    return features


def sent2features(sent):
    return [word2features(sent, i)
            for i in range(len(sent))]

def train_ner(nlu_data: dict, verbose: bool = False):
    if verbose: print_log('[train_ner] loading training data')
    ent_data = []
    for data in nlu_data['data']:
        ent_data.extend(data['examples'])

    bio_data = ent2bio(ent_data)
    x_train = [
        sent2features(tokens)
        for tokens, labels in bio_data
    ]
    y_train = [
        labels
        for tokens, labels in bio_data
    ]

    if verbose: print_log('[train_ner] training ner model')
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(x_train, y_train)
    if verbose: print_log(f'train_ner] ner model: {crf}')

    return crf

def predict_ner(ner_model, text) -> list:
    tokens = text.split()
    features = sent2features(tokens)
    pred = ner_model.predict([features])[0]
    return bio2ent(tokens, pred)
