
import streamlit
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datasets
import pathlib
from streamlit.uploaded_file_manager import UploadedFile
import tensorflow as tf
import tensorflow_hub
from whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage, SpacyLanguage
from whatlies.language import TFHubLanguage
from whatlies import Embedding, EmbeddingSet
from whatlies.transformers import Pca, Umap, Tsne, Lda
from sentence_transformers import SentenceTransformer
from preshed.maps import PreshMap
from cymem.cymem import Pool
import json


from bulk_labelling.load_config import load_languages,load_transformer,load_dataset
from bulk_labelling.embedding import get_embedding,get_embeddingset,get_language_array
from bulk_labelling.plotting import prepare_data,make_plot


def write():
    Embedding_frameworks = pd.Series(['Byte Pair Language', 'English Spacy', 'Multilingual Universal Sentence encoder', 'Distilbert Multilingual', 'French Spacy', 'Universal Sentence Encoder', 'CountVectorLanguage', 'flauBERT', 'camemBERT'])
    Embedding_frameworks_languages = pd.Series(['english', 'english', 'multilingual', 'multilingual', 'french', 'english', 'english', 'french', 'french'])
    Embedding_frameworks_dataframe = pd.DataFrame(Embedding_frameworks)
    Embedding_frameworks_dataframe['language'] = Embedding_frameworks_languages
    Embedding_frameworks_dataframe.columns = ['framework', 'language']


    # @streamlit.cache
    # def get_embedding(vec, text):
    #     return Embedding(text, vec)


    # @streamlit.cache
    # def get_embeddingset(veclist, textlist):
    #     return EmbeddingSet(*[get_embedding(veclist[q], textlist[q]) for q in range(len(textlist))])


    # @streamlit.cache
    # def get_language_array(lang, textlist=None):
    #     if isinstance(lang, EmbeddingSet):
    #         return lang.to_names_X()[1], lang.to_names_X()[0]
    #     if isinstance(lang, SentenceTransformer):
    #         encoding = lang.encode(textlist)
    #         # streamlit.write(encoding.shape)
    #         return encoding, textlist
    #     else:
    #         return lang[textlist].to_names_X()[1], lang[textlist].to_names_X()[0]


    # @streamlit.cache
    # def prepare_data(lang, transformer, textlist=None):
    #     progress_container.text("preparing data...")

    #     encoding, texts = get_language_array(lang, textlist)
    #     embset = get_embeddingset(encoding, texts)
    #     result = embset.transform(transformer)
    #     progress_container.text("data prepared!")
    #     return result


    # @streamlit.cache
    # def make_plot(lang, transformer, textlist=None):
    #     return prepare_data(lang, transformer, textlist).plot_interactive(annot=False).properties(width=1000, height=500, title=type(lang).__name__)


    datasets_dict = ["-", "bing_coronavirus_query_set", "app_reviews", "emotion"]
    transformers_dict = {'PCA': "Pca(2)", 'TSNE': "Tsne(2)", 'Umap': "Umap(2)"}
    languages_dict = {'CountVectorLanguage': "CountVectorLanguage(10)",
                    'Universal Sentence Encoder': "TFHubLanguage(os.path.abspath('data/models/universal-sentence-encoder_4'))",
                    'Byte Pair Language': "BytePairLanguage('en', dim=300, vs=200_000)",
                    'Multilingual Universal Sentence encoder': "TFHubLanguage(os.path.abspath('data/models/universal-sentence-encoder-multilingual-large_3'))",
                    'Distilbert Multilingual': "SentenceTransformer('quora-distilbert-multilingual')",
                    'French Spacy': "SpacyLanguage('fr_core_news_sm')",
                    'English Spacy': "SpacyLanguage('en_core_web_md')",
                    'FlauBERT': "SentenceTransformer('flaubert/flaubert_base_uncased')",
                    'camemBERT': "SentenceTransformer('camembert-base')"
                    }


    streamlit.sidebar.title('Bulk labelling')

    dataset_upload = streamlit.sidebar.beta_expander('1. Select your dataset')
    available_datasets = datasets_dict + [i for i in os.listdir('data/datasets') if '.csv' in i]
    option = dataset_upload.selectbox('Dataset:', available_datasets, index=0)
    progress_container = streamlit.empty()


    # @streamlit.cache
    # def load_dataset(dataset_name,datasets_dict):
    #     progress_container.text("loading dataset...")
    #     if dataset_name in datasets_dict:
    #         if dataset_name != '-':
    #             if dataset_name == 'bing_coronavirus_query_set':
    #                 dataset = datasets.load_dataset("bing_coronavirus_query_set", queries_by="country", start_date="2020-09-01", end_date="2020-09-30")
    #             else:
    #                 dataset = datasets.load_dataset(dataset_name)
    #             dataset = pd.DataFrame.from_dict(dataset['train'])
    #             progress_container.text("dataset loaded!")
    #             return dataset
    #     else:
    #         dataset = pd.read_csv(f'data/datasets/{dataset_name}')
    #         progress_container.text("dataset loaded!")

    #         return dataset


    dataset = load_dataset(option,datasets_dict)
    dataframe_preview = dataset_upload.empty()
    uploaded_file = dataset_upload.file_uploader("Add a custom dataset")
    uploaded_file_name = dataset_upload.text_input("Custom dataset name")


    try:
        dataframe_preview.dataframe(dataset.head())
    except Exception:
        pass

    if (uploaded_file is not None) and (uploaded_file_name is not None):
        uploaded_dataset = pd.read_csv(uploaded_file)
        uploaded_dataset.to_csv('data/datasets/{}.csv'.format(uploaded_file_name), index=False)

    embedding = streamlit.beta_container()
    embedding_lang_select, embedding_lang = embedding.beta_columns(2)
    languages_embedding = embedding_lang_select.multiselect('Embedding framework languages', ['english', 'french', 'multilingual'], ['english', 'french', 'multilingual'])
    embedding_language = embedding_lang.selectbox('Embedding framework', Embedding_frameworks_dataframe[Embedding_frameworks_dataframe.language.isin(languages_embedding)].framework.tolist())
    lang = load_languages(embedding_language, languages_dict)


    map = streamlit.beta_container()
    transformer_option = map.selectbox('Dimension reduction framework', ('TSNE', 'PCA', 'Umap'))
    transformer = load_transformer(transformer_option,transformers_dict)


    try:
        column_name = streamlit.selectbox('columns', options=['-'] + dataset.columns.tolist())

        if column_name != '-':
            streamlit.altair_chart(make_plot(lang, transformer, dataset[column_name].astype(str).head(5000).tolist()), use_container_width=True)
    except Exception:
        pass
