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
    embedding_df.loc[embedding_df.labelling_uuid.isin(
        temp_embedding_df[temp_embedding_df.labels != 'None'].labelling_uuid), 'labels'] = label

    return embedding_df


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


def remove_stopwords_from_column(Textcolumn, dataset_language):
    if dataset_language == 'English':
        nlp = spacy.load('en_core_web_md')
    if dataset_language == 'French':
        nlp = spacy.load('fr_core_news_sm')
    result = Textcolumn.apply(lambda text:
                              " ".join(token.lemma_ for token in nlp(text)
                                       if not token.is_stop))

    return result

def cluster_tfidf(df,language='French'):
    if language=='English':
        stopwords_list=en_stop
    if language == 'French':
        stopwords_list=fr_stop

    for k in df.cluster.unique():
        wordcloud = WordCloud(stopwords=stopwords_list, background_color="white",
                            colormap='Blues',width = 100, height = 100).generate(' '.join(df[df.cluster==k].text.tolist()).lower())
        top_3 = list(wordcloud.words_.keys())
        # streamlit.write(df[df.cluster==k])
        # streamlit.write(f'{top_3[0]},{top_3[1]},{top_3[2]}')
        df.loc[df.cluster==k,'cluster'] =  f'{top_3[0]} - {top_3[1]} - {top_3[2]}'
    return df
    
