
from datasets import dataset_dict
from sklearn import cluster
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
from whatlies import language
from whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage, SpacyLanguage
from whatlies.language import TFHubLanguage
from whatlies import Embedding, EmbeddingSet
from whatlies.transformers import Pca, Umap, Tsne, Lda
from sentence_transformers import SentenceTransformer
from preshed.maps import PreshMap
from cymem.cymem import Pool
import json
from hulearn.preprocessing import InteractivePreprocessor
from streamlit_bokeh_events import streamlit_bokeh_events
from bokeh.models import DataTable, TableColumn, ColumnDataSource, CustomJS
from sklearn.cluster import OPTICS


from bulk_labelling.load_config import load_languages, load_transformer, load_dataset, load_config
from bulk_labelling.embedding import get_embedding, get_embeddingset, get_language_array
from bulk_labelling.plotting import prepare_data, make_plot, make_interactive_plot, suggest_clusters

from bulk_labelling.pages import plain_plot, cluster_suggestion


def write():

    languages_dict, transformers_dict, datasets_dict, Embedding_frameworks_dataframe = load_config()

    streamlit.sidebar.title('Bulk labelling')
    dataset_upload = streamlit.sidebar.beta_expander('1. Select your dataset')

    available_datasets = datasets_dict + \
        [i for i in os.listdir('data/datasets') if '.csv' in i]
    option = dataset_upload.selectbox('Dataset:', available_datasets, index=0)

    dataset = None
    dataframe_preview = dataset_upload.empty()
    uploaded_file = dataset_upload.file_uploader("Add a custom dataset")
    uploaded_file_name = dataset_upload.text_input("Custom dataset name")
    dataset = load_dataset(option, datasets_dict)
    try:
        dataframe_preview.dataframe(dataset.head())
    except Exception:
        pass

    if (uploaded_file is not None) and (uploaded_file_name is not None):
        uploaded_dataset = pd.read_csv(uploaded_file)
        uploaded_dataset.to_csv(
            'data/datasets/{}.csv'.format(uploaded_file_name), index=False)

    # interactive = streamlit.sidebar.checkbox('Draw interactive chart')
    # suggest_clusters = streamlit.sidebar.checkbox('Suggest clusters')

    embedding = streamlit.sidebar.beta_expander(
        "2. Select your embedding framework")
    embedding_lang_select = embedding.beta_container()
    embedding_lang = embedding.beta_container()
    languages_embedding = embedding_lang_select.multiselect('Embedding framework languages', [
                                                            'english', 'french', 'multilingual'], ['english', 'french', 'multilingual'])
    embedding_language = embedding_lang.selectbox(
        'Embedding framework', Embedding_frameworks_dataframe[Embedding_frameworks_dataframe.language.isin(languages_embedding)].framework.tolist())

    map = streamlit.sidebar.beta_expander("3. Select your DR algorithm")
    transformer_option = map.selectbox(
        'Dimension reduction framework', ('TSNE', 'PCA', 'Umap'))

    chart_container, options_container = streamlit.beta_columns(2)
    info_container = options_container.empty()
    dataview_container = options_container.empty()
    name_select = options_container.empty()
    name_clear = options_container.empty()

    plot = None
    df = None
    column_name = '-'
    embedding_df = None

    column_select = streamlit.sidebar.beta_expander(
        '4. Select column for analysis')

    graph_type = streamlit.sidebar.radio("How would you like to display the graph?",
                                         ('Classic viz', 'Interactive labeling', 'cluster suggestion'))

    space = streamlit.sidebar.beta_container()

    validate = streamlit.sidebar.checkbox('refresh plot')

    clear_cache = streamlit.sidebar.button('clear cache')

    if clear_cache:
        [f.unlink() for f in pathlib.Path("data/plotting_data").glob("*") if f.is_file()]

    if dataset is not None:
        column_name = column_select.selectbox(
            'columns', options=['-'] + dataset.columns.tolist())

    if validate and column_name != '-':

        if ('cache.json' not in os.listdir('data/plotting_data')):
            lang = load_languages(embedding_language, languages_dict)
            transformer = load_transformer(
                transformer_option, transformers_dict)
            textlist = dataset.sample(frac=1)[column_name].astype(str).head(5000).tolist()
            embset = prepare_data(lang, transformer, textlist)
            embedding_df = embset.to_dataframe().reset_index()
            embedding_df.to_csv('data/plotting_data/cache.csv', index=False)
            json_cache = {'dataset': option, 'column': column_name,
                          'language_model': embedding_language, 'reduction_algorithm': transformer_option}
            with open('data/plotting_data/cache.json', 'w', encoding='utf-8') as f:
                json.dump(json_cache, f, ensure_ascii=False, indent=4)

        if ('cache.json' in os.listdir('data/plotting_data')):
            f = open('data/plotting_data/cache.json')
            cached_data = json.load(f)
            json_cache = {'dataset': option, 'column': column_name,
                          'language_model': embedding_language, 'reduction_algorithm': transformer_option}
            if cached_data != json_cache:
                lang = load_languages(embedding_language, languages_dict)
                transformer = load_transformer(
                    transformer_option, transformers_dict)
                textlist = dataset.sample(frac=1)[column_name].astype(str).head(5000).tolist()
                embset = prepare_data(lang, transformer, textlist)
                embedding_df = embset.to_dataframe().reset_index()
                embedding_df.to_csv(
                    'data/plotting_data/cache.csv', index=False)
                json_cache = {'dataset': option, 'column': column_name,
                              'language_model': embedding_language, 'reduction_algorithm': transformer_option}
                with open('data/plotting_data/cache.json', 'w', encoding='utf-8') as f:
                    json.dump(json_cache, f, ensure_ascii=False, indent=4)
            else:
                embedding_df = pd.read_csv('data/plotting_data/cache.csv')
        embedding_df = embedding_df.rename(columns={
                                           embedding_df.columns[0]: 'text', embedding_df.columns[1]: 'd1', embedding_df.columns[2]: 'd2'})
        if 'labels' not in embedding_df.columns:
            embedding_df['labels'] = ''
        if embedding_df is not None:

            if graph_type == 'Interactive labeling':

                plot, df = make_interactive_plot(embedding_df)
                with chart_container:
                    result_lasso = streamlit_bokeh_events(
                        bokeh_plot=plot,
                        events="LASSO_SELECT",
                        key="bar",
                        refresh_on_update=True,
                        debounce_time=10)

                    if result_lasso:
                        if result_lasso.get("LASSO_SELECT"):
                            selected_data = df.iloc[result_lasso.get("LASSO_SELECT")[
                                "data"]]
                            info_container.info(
                                f'selected {len(selected_data)} rows')
                            dataview_container.write(selected_data.text)
                            name_select_value = name_select.text_input(
                                'Input selected data label')
                            if name_select_value:
                                selected_data['label'] = name_select_value
                                selected_data.to_csv(
                                    'data/labeled_data/test_data.csv', encoding='utf-8', index=False)

            if graph_type == 'Classic viz':
                plain_plot.write(embedding_df, dataset)

            if graph_type == 'cluster suggestion':
                cluster_suggestion.write(embedding_df)
