
from datasets import dataset_dict
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


from bulk_labelling.load_config import load_languages, load_transformer, load_dataset, load_config
from bulk_labelling.embedding import get_embedding, get_embeddingset, get_language_array
from bulk_labelling.plotting import prepare_data, make_plot, make_interactive_plot


def write():

    languages_dict, transformers_dict, datasets_dict, Embedding_frameworks_dataframe = load_config()

    streamlit.sidebar.title('Bulk labelling')

    dataset_upload = streamlit.sidebar.beta_expander('1. Select your dataset')
    available_datasets = datasets_dict + \
        [i for i in os.listdir('data/datasets') if '.csv' in i]
    option = dataset_upload.selectbox('Dataset:', available_datasets, index=0)

    column_select = streamlit.sidebar.beta_expander(
        '2. Select column for analysis')

    dataset = load_dataset(option, datasets_dict)
    dataframe_preview = dataset_upload.empty()
    uploaded_file = dataset_upload.file_uploader("Add a custom dataset")
    uploaded_file_name = dataset_upload.text_input("Custom dataset name")

    interactive = streamlit.sidebar.checkbox('Draw interactive chart')

    try:
        dataframe_preview.dataframe(dataset.head())
    except Exception:
        pass

    if (uploaded_file is not None) and (uploaded_file_name is not None):
        uploaded_dataset = pd.read_csv(uploaded_file)
        uploaded_dataset.to_csv(
            'data/datasets/{}.csv'.format(uploaded_file_name), index=False)

    embedding = streamlit.beta_container()
    embedding_lang_select, embedding_lang = embedding.beta_columns(2)
    languages_embedding = embedding_lang_select.multiselect('Embedding framework languages', [
                                                            'english', 'french', 'multilingual'], ['english', 'french', 'multilingual'])
    embedding_language = embedding_lang.selectbox(
        'Embedding framework', Embedding_frameworks_dataframe[Embedding_frameworks_dataframe.language.isin(languages_embedding)].framework.tolist())
    lang = load_languages(embedding_language, languages_dict)

    map = streamlit.beta_container()
    transformer_option = map.selectbox(
        'Dimension reduction framework', ('TSNE', 'PCA', 'Umap'))
    transformer = load_transformer(transformer_option, transformers_dict)

    chart_container, options_container = streamlit.beta_columns(2)
    info_container=options_container.empty()
    dataview_container=options_container.empty()
    name_select=options_container.empty()
    plot=None
    df=None
    try:
        column_name = column_select.selectbox(
            'columns', options=['-'] + dataset.columns.tolist())

        if column_name != '-':
            if interactive:
                
                plot, df = make_interactive_plot(
                    lang, transformer, dataset[column_name].astype(str).head(5000).tolist())
                with chart_container:
                    result_lasso = streamlit_bokeh_events(
                        bokeh_plot=plot,
                        events="LASSO_SELECT",
                        key="bar",
                        refresh_on_update=True,
                        debounce_time=10)
                    
                    if result_lasso:
                        if result_lasso.get("LASSO_SELECT"):
                            selected_data=df.iloc[result_lasso.get("LASSO_SELECT")["data"]]['text']
                            info_container.info(f'selected {len(selected_data)} rows')
                            dataview_container.write(selected_data)
                            name_select_value=name_select.text_input('Input selected data label')
                
            else:
                chart = make_plot(lang, transformer, dataset[column_name].astype(
                    str).head(5000).tolist())
                chart_container.altair_chart(chart, use_container_width=True)
                options_container.info(f'looking at {len(dataset)} examples')
    except Exception:
        chart = None
