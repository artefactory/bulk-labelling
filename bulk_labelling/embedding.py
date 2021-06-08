
import streamlit
from bulk_labelling.custom_whatlies.embedding import Embedding
from bulk_labelling.custom_whatlies.embeddingset import EmbeddingSet
from sentence_transformers import SentenceTransformer


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


def cluster(algo, embset):
    """suggests cluster based on provided algo and embedding dataframe

    Args:
        algo (sklearn.transformer): clustering algorithm used to cluster data
        embset (pd.DataFrame): embedding dataframe containing x,y data to cluster

    Returns:
        pd.dataFrame(): embedding dataframe with cluster suggestion values
    """
    transformed = embset[['d1', 'd2']].values
    clustering = algo.fit(transformed)
    labels = clustering.labels_
    embed = embset.copy()
    embed['cluster'] = labels
    embed['cluster'] = embed['cluster'].astype(str)
    return embed
