import pickle
import json
from datetime import datetime

from util.logger import print_log
from .entity_tagger import train_ner, sent2features, bio2ent
from .intent_classifier import train_intent

class NLUComponent:
    def __init__(self, timestamp, retrain_model: bool=False):
        self.ner_model = None
        self.intent_model = None

        if retrain_model:
            train(verbose=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            self.intent_model, self.ner_model = load_nlu_model(date_str)

        else:
            self.intent_model, self.ner_model = load_nlu_model(timestamp)

    def predict_intent(self, text):
        return self.intent_model.predict([text])[0]

    def predict_ner(self, text) -> list:
        tokens = text.split()
        features = sent2features(tokens)
        pred = self.ner_model.predict([features])[0]
        return bio2ent(tokens, pred)


def _load_nlu_data(data_path):
    training_data = None
    with open(data_path, 'r', encoding='utf-8') as file:
        training_data = json.load(file)
    return training_data

def train(verbose: bool=False):
    print_log('[nlu_trainer] start nlu training')
    training_data = _load_nlu_data('data/nlu_data.json')
    intent_model = train_intent(training_data, verbose=verbose)
    ner_model = train_ner(training_data, verbose=verbose)

    date_str = datetime.now().strftime("%Y-%m-%d")
    if verbose: print_log('[nlu_trainer] saving model')
    with open(f'model/intent_{date_str}.pkl', 'wb') as file:
        pickle.dump(intent_model, file)

    with open(f'model/ner_{date_str}.pkl', 'wb') as file:
        pickle.dump(ner_model, file)

    print_log('[nlu_trainer] training nlu complete')

def load_nlu_model(timestamp):
    print_log(f'[nlu_trainer] loading nlu model from {timestamp}')
    with open(f'model/intent_{timestamp}.pkl', 'rb') as file:
        intent_model = pickle.load(file)
    with open(f'model/ner_{timestamp}.pkl', 'rb') as file:
        ner_model = pickle.load(file)

    return intent_model, ner_model



if __name__ == '__main__':
    train(verbose=True)