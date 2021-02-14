# flake8: noqa

import random
import string

import nltk
import numpy as np
from nltk.tokenize import wordpunct_tokenize

from adthena_task.config import Config

config = Config()


def clean_text(text: str, stemmer: nltk.stem.api.StemmerI, stop_words) -> str:
    """
    Performs basic cleaning operations along with stemming, using nltk Stemmer.
    Args:
        text: Text to be processed.
        stemmer: Stemmer from nltk api.

    Returns:
        Processed text.
    """
    text = str(text).lower()
    tokens = [
        stemmer.stem(word)
        for word in wordpunct_tokenize(text)
        if word not in list(stop_words) + list(string.punctuation)
    ]
    text = " ".join(tokens)
    return text


def tokenize_text(text: str) -> str:
    """
    Performs text tokenization
    Args:
        text:

    Returns:

    """
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


def create_embedding_matrix(word_index, embedding_dict, dimension):
    embedding_matrix = np.zeros((len(word_index) + 1, dimension))

    for word, index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
    return embedding_matrix


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    # Reinitializing random value, problem described for example in:
    # https://stackoverflow.com/questions/44443675/removing-randomization-of-vector-initialization-for-doc2vec
    random.seed(config.SEED)
    targets, regressors = zip(
        *[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents]
    )
    return targets, regressors
