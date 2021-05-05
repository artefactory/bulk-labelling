
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


Embedding_frameworks = pd.Series(['Byte Pair Language', 'English Spacy','Multilingual Universal Sentence encoder', 'Distilbert Multilingual', 'French Spacy', 'Universal Sentence Encoder','CountVectorLanguage','flauBERT','camemBERT'])
Embedding_frameworks_languages = pd.Series(['english', 'english', 'multilingual', 'multilingual', 'french', 'english', 'english','french','french'])
Embedding_frameworks_dataframe = pd.DataFrame(Embedding_frameworks)
Embedding_frameworks_dataframe['language'] = Embedding_frameworks_languages
Embedding_frameworks_dataframe.columns = ['framework', 'language']


@streamlit.cache
def get_embedding(vec, text):
    return Embedding(text, vec)

@streamlit.cache
def get_embeddingset(veclist, textlist):
    return EmbeddingSet(*[get_embedding(veclist[q], textlist[q]) for q in range(len(textlist))])


@streamlit.cache
def get_language_array(lang,textlist=None):
    if isinstance(lang, EmbeddingSet):
        return lang.to_names_X()[1],lang.to_names_X()[0]
    if isinstance(lang,SentenceTransformer):
        encoding=lang.encode(textlist)
        # streamlit.write(encoding.shape)
        return encoding,textlist
    else:
        return lang[textlist].to_names_X()[1],lang[textlist].to_names_X()[0]
    

@streamlit.cache
def prepare_data(lang, transformer, textlist=None):
    progress_container.text("preparing data...")

    encoding,texts= get_language_array(lang,textlist)   
    embset=get_embeddingset(encoding,texts)
    result=embset.transform(transformer)
    # streamlit.write(result)
    # progress_container.text("preparing data...")
    # streamlit.write(f'length of textlist :{len(textlist)}')
    # if isinstance(lang,SentenceTransformer):
    #     encoding=lang.encode(textlist)
    #     streamlit.write(f'length of encoding by distilbert: {len(encoding)}')
    #     result = get_embeddingset(lang.encode(textlist),textlist).transform(transformer)
    # else:
        # result = lang[textlist].transform(transformer)
    progress_container.text("data prepared!")
    return result


@streamlit.cache
def make_plot(lang, transformer, textlist=None):
    return prepare_data(lang, transformer, textlist).plot_interactive(annot=False).properties(width=1000, height=500, title=type(lang).__name__)

streamlit.write()


def load_languages(language):
    if language=='CountVectorLanguage':
        return CountVectorLanguage(10)
    if language=='Universal Sentence Encoder':
        return TFHubLanguage(os.path.abspath('universal-sentence-encoder_4'))
    if language=='Byte Pair Language':
        return BytePairLanguage("en", dim=300, vs=200_000)
    if language=='Multilingual Universal Sentence encoder':
        return TFHubLanguage(os.path.abspath('universal-sentence-encoder-multilingual-large_3'))
    if language == 'Distilbert Multilingual':
        return SentenceTransformer('quora-distilbert-multilingual')
    if language == 'French Spacy':
        return SpacyLanguage('fr_core_news_sm')
    if language == 'English Spacy':
        return SpacyLanguage("en_core_web_md")
    if language == 'FlauBERT':
        return SentenceTransformer('flaubert/flaubert_base_uncased')
    if language == 'camemBERT':
        return SentenceTransformer('camembert-base')
    return CountVectorLanguage(10)


datasets_dict = {"-": pd.DataFrame(), "bing_coronavirus_query_set": pd.DataFrame(), "app_reviews": pd.DataFrame(), "emotion": pd.DataFrame()}
transformers_dict = {'PCA': Pca(2), 'TSNE': Tsne(2), 'Umap': Umap(2)}





streamlit.title('Bulk labelling')

# streamlit.sidebar.title('Dataset parameters')
# languages = streamlit.sidebar.multiselect('Dataset languages', ['english', 'french', 'multilingual'], ['english', 'french', 'multilingual'])

selectbox_container = streamlit.sidebar.empty()
available_datasets=list(datasets_dict.keys())+[i for i in os.listdir() if '.csv' in i]
option = selectbox_container.selectbox('Dataset:', available_datasets, index=0)
progress_container=streamlit.empty()

@streamlit.cache
def load_dataset(dataset_name):
    progress_container.text("loading dataset...")
    if dataset_name in list(datasets_dict.keys()):
        if dataset_name != '-':
            if dataset_name == 'bing_coronavirus_query_set':
                dataset = datasets.load_dataset("bing_coronavirus_query_set", queries_by="country", start_date="2020-09-01", end_date="2020-09-30")
            else:
                dataset = datasets.load_dataset(dataset_name)
            dataset = pd.DataFrame.from_dict(dataset['train'])
            datasets_dict[dataset_name] = dataset
            progress_container.text("dataset loaded!")
            return dataset
    else:
        dataset=pd.read_csv(dataset_name)
        progress_container.text("dataset loaded!")

        return dataset
    
dataframe_preview=streamlit.sidebar.empty()

dataset = load_dataset(option)
uploaded_file = streamlit.sidebar.file_uploader("Add a custom dataset")
uploaded_file_name= streamlit.sidebar.text_input("Custom dataset name")

try:
    dataframe_preview.dataframe(dataset.head())
except:
    pass

if uploaded_file!=None and uploaded_file_name!=None:
    uploaded_dataset=pd.read_csv(uploaded_file)
    uploaded_dataset.to_csv('{}.csv'.format(uploaded_file_name),index=False)

embedding, map = streamlit.beta_columns(2)
embedding.header("Embedding")
languages_embedding = embedding.multiselect('Embedding framework languages', ['english', 'french', 'multilingual'], ['english', 'french', 'multilingual'])
embedding_language = embedding.selectbox('Embedding framework', Embedding_frameworks_dataframe[Embedding_frameworks_dataframe.language.isin(languages_embedding)].framework.tolist())
lang = load_languages(embedding_language)


map.header("Dimension reduction")

transformer_option = map.selectbox('Dimension reduction framework', ('TSNE', 'PCA', 'Umap'))
transformer = transformers_dict[transformer_option]
# map.write(lang)
# map.write(transformer)
try:
    column_name = streamlit.selectbox('columns',options=['-']+dataset.columns.tolist())
    
    if column_name!='-':
        streamlit.altair_chart(make_plot(lang, transformer, dataset[column_name].astype(str).head(5000).tolist()),use_container_width=True)
except Exception as error:
    pass
# slidernumber = streamlit.slider('number of points', 0, 2000, 1000)

