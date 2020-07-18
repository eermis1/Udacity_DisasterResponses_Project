import sys
import pandas as pd
import numpy as np
import os
import pickle
from sqlalchemy import create_engine
import re
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
import warnings


def load_data(database_filepath):

    """
    - Loads data from SQL Database

    Args:
    database_filepath: SQL database file

    Returns:
    X pandas_dataframe: Features dataframe
    Y pandas_dataframe: Target dataframe
    category_names list: Target labels
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('clean_dataset', engine)
    X = df["message"]
    Y = df[df.columns[4:]]
    category_names = df.columns[4:]

    return X, Y, category_names


def tokenize(text):

    """
    - Remove capitalization and special characters and lemmatize texts

    Args:
    messages as text file

    Returns:
    Processed text after normalizing, tokenizing and lemmatizing
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    """
    - Initialize Verb Extractor class

    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier.
    It has been integrated from Machine Learning Pipeline - Solution: GridSearch
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():

    """
    - Build model with GridSearch

    Returns:
    Trained model with GridSearch
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # number of parameter is limited with 2 because of high computation duration
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #'features__text_pipeline__tfidf__use_idf': (True, False),
    }

    warnings.filterwarnings("ignore")
    pipeline = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    """
    - Illustrates model performance based on test data

    Args:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    """

    Y_predictions = model.predict(X_test)

    accuracy = (Y_predictions == Y_test).mean().mean()
    print('\n Overall model accuracy is:  {0:.2f}% \n'.format(accuracy * 100))

    Y_predictions_df = pd.DataFrame(Y_predictions, columns=Y_test.columns)

    # Print classification report
    # to limit report length, number of colums are limited with 10
    i = 0
    for column in Y_test.columns:
        i = i + 1
        if (i <= 10):
            print('....................................................\n')
            print('FEATURE: {}\n'.format(column))
            print(classification_report(Y_test[column], Y_predictions_df[column]))


def save_model(model, model_filepath):

    """
    - Saves the model as Python Pickle file.

    Args:
    model: Trained model
    model_filepath: Filepath to save the model
    """

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        warnings.filterwarnings("ignore")

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
