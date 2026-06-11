import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def train_intent_model(data: dict, verbose: bool):
    model_name = data["model"]
    model = None
    x_train = []
    y_train = []

    for sample in data["data"]:
        x_train.extend(sample["examples"])
        y_train.extend([sample["intent"]] * len(sample["examples"]))

        if len(x_train) != len(y_train):
            print("NOT SAME!!!", len(x_train), len(y_train))
            return

    if verbose:
        print("total data", len(x_train))

    if model_name == "lsvc":
        if verbose:
            print("train lsvc")
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2
            )),
            ("clf", SVC(kernel="linear", probability=True, random_state=42))
        ])
        model.fit(x_train, y_train)

    elif model_name == "lgs":
        if verbose:
            print("train lgs")
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2
            )),
            ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs"))
        ])
        model.fit(x_train, y_train)

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
