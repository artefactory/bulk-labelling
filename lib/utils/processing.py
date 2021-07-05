from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# from nltk.tokenize import word_tokenize
import streamlit
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop


def replace_labels(embedding_df, temp_embedding_df, label):
    """replaces the label in a given part of the embedding dataframe

    Args:
        embedding_df (pd.DataFrame): dataframe of dimension-reduced encodings for text
        temp_embedding_df (pd.DataFrame): temporary dataframe with the selected data and/or clustering
        label (str): label to replace the labels in embedding_df with

    Returns:
        pd.DataFrame(): dataframe with replaced labels where necessary
    """
    embedding_df.loc[
        embedding_df.labelling_uuid.isin(
            temp_embedding_df[temp_embedding_df.labels != "None"].labelling_uuid
        ),
        "labels",
    ] = label

    return embedding_df


def cluster(algo, embset):
    """suggests cluster based on provided algo and embedding dataframe

    Args:
        algo (sklearn.transformer): clustering algorithm used to cluster data
        embset (pd.DataFrame): embedding dataframe containing x,y data to cluster

    Returns:
        pd.dataFrame(): embedding dataframe with cluster suggestion values
    """
    transformed = embset[["d1", "d2"]].values
    clustering = algo.fit(transformed)
    labels = clustering.labels_
    embed = embset.copy()
    embed["cluster"] = labels
    embed["cluster"] = embed["cluster"].astype(str)
    return embed


def remove_stopwords_from_column(Textcolumn, dataset_language, custom_stopwords):
    """Removes stopwords from the text column of the dataframe

    Args:
        Textcolumn (pd.Series): pandas Series description of the column to remove stopwords from
        dataset_language (str): language of model to load from spacy to get default stopwords
        custom_stopwords (list): list of custom stopwords to add to default language stopwords

    Returns:
        pd.Series : modified text column with stopwords removed
    """
    if dataset_language == "English":
        nlp = spacy.load("en_core_web_md")
    if dataset_language == "French":
        nlp = spacy.load("fr_core_news_sm")
    if custom_stopwords is not None:
        for k in custom_stopwords:
            nlp.Defaults.stop_words.add(k)
    result = Textcolumn.apply(
        lambda text: " ".join(token.text for token in nlp(text) if not token.is_stop)
    )

    return result


def cluster_tfidf(df, language="French"):
    """generates an estimation of possible topic names for suggested clusters

    Args:
        df (pd.DataFrame): dataframe with cluster suggestion numbers
        language (str, optional): language to load spacy model for stopwords from. Defaults to "French".

    Returns:
        pd.DataFrame : modified dataframe with cluster numbers replaced with top 3 cluster name suggestions
    """
    if language == "English":
        stopwords_list = en_stop
    if language == "French":
        stopwords_list = fr_stop

    for k in df.cluster.unique():
        wordcloud = WordCloud(
            stopwords=stopwords_list,
            background_color="white",
            colormap="Blues",
            width=100,
            height=100,
        ).generate(" ".join(df[df.cluster == k].text.astype(str).tolist()).lower())
        top_3 = list(wordcloud.words_.keys())
        df.loc[df.cluster == k, "cluster"] = f"{top_3[0]} - {top_3[1]} - {top_3[2]}"
    return df
