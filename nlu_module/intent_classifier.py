import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import re

from util.logger import print_log

def model_selector(model_name):
    model = None
    if model_name == "lsvc":
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2
            )),
            ("clf", SVC(kernel="linear", probability=True, random_state=42))
        ])


    elif model_name == "lgs":
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2
            )),
            ("clf", LogisticRegression(solver="lbfgs"))
        ])

    return model

def train_intent(data: dict, verbose: bool):
    model_name = data["model"] if data["model"] != 'default' else 'lgs'
    model = None
    x_train = []
    y_train = []

    for sample in data["data"]:
        clean_text  = [re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text) for text in sample["examples"]]
        x_train.extend(clean_text)
        y_train.extend([sample["intent"]] * len(clean_text))

        if len(x_train) != len(y_train):
            if verbose: print_log('[train_intent] ERROR! diff len x_train and y_train')
            return None

    if verbose: print_log(f'[train_intent] train {model_name}')
    model = model_selector(model_name)
    model.fit(x_train, y_train)

    if verbose: print_log(f'train_intent] intent model: {model}')

    return model


def eval_intent_model(model, data: dict, return_proba=False):
    for sample in data:
        print("#" * 10, f'test intent: {sample["intent"]}', "#" * 10)
        for text_sample in sample["examples"]:
            if return_proba:
                intent = model.predict_proba([text_sample])[0]
                intent = np.round(intent, 3)
                score = " | ".join([
                    f"{c[0]}:{c[1]}"
                    for c in zip(list(model.classes_), intent)
                ])
                print(f"\t{text_sample} -> {score}")

            else:
                intent = model.predict([text_sample])[0]
                print(f"\t{text_sample} -> {intent}")
