
import streamlit
from lib.custom_whatlies.embedding import Embedding
from lib.custom_whatlies.embeddingset import EmbeddingSet
from sentence_transformers import SentenceTransformer
import logging
import time


@streamlit.cache
def get_embedding(vec, text):
    """gets a single whatlies embedding from a textlist and a vector

    Args:
        vec (numpy.array): embedding vector for text
        text (str): text to embed

    Returns:
        whatlies.Embedding: embedding constructed with the given vector and text
    """
    return Embedding(text, vec)


@streamlit.cache
def get_embeddingset(veclist, textlist):
    """gets a whatlies.embeddingset from the encoding given by the language model

    Args:
        veclist (numpy.ndarray): ndarray of all encodings
        textlist (list): vector of encoded texts

    Returns:
        whatlies.EmbeddingSet: whatlies EmbeddingSet for easier transformation
    """

    return EmbeddingSet(*[get_embedding(veclist[q], textlist[q]) for q in range(len(textlist))])


def get_language_array(lang, textlist=None):
    """gets language array with encoding and texts for all possible models as an EmbeddingSet

    Args:
        lang (whatlies, tfhub or huggingface model): language model to encode texts
        textlist (list, optional): list of texts to encode. Defaults to None.

    Returns:
        list: list with the first elements being the encoding of the texts and the second being the texts
    """
    if isinstance(lang, EmbeddingSet):
        return lang.to_names_X()[1], lang.to_names_X()[0]
    if isinstance(lang, SentenceTransformer):
        encoding = lang.encode(textlist)
        return encoding, textlist
    else:
        return lang[textlist].to_names_X()[1], lang[textlist].to_names_X()[0]


def prepare_data(lang, textlist=None):
    """encodes and transforms a list of texts with the models chosen by the user

    Args:
        lang (huggingface transformer, tfhub language or whatlies language): model used to encode the texts
        transformer (whatlies transformer): dimension reduction transformer chosn by the user
        textlist (list, optional): list of texts to encode. Defaults to None.

    Returns:
        pd.DataFrame: dataframe containing the texts to encode as well as their n-D transformed encodings
    """
    start = time.time()
    encoding, texts = get_language_array(lang, textlist)
    step1 = time.time()
    embset = get_embeddingset(encoding, texts)
    step2 = time.time()
    end = time.time()
    logging.info(f'getting encodings : {step1-start}')
    logging.info(f'getting embeddingset : {step2-step1}')

    return embset
