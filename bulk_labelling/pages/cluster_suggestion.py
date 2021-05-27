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
# from bulk_labelling.custom_whatlies import language
from bulk_labelling.custom_whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage, SpacyLanguage
from bulk_labelling.custom_whatlies.language import TFHubLanguage
from bulk_labelling.custom_whatlies.embedding import Embedding
from bulk_labelling.custom_whatlies.embeddingset import EmbeddingSet
from bulk_labelling.custom_whatlies.transformers import Pca, Umap, Tsne
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
from bulk_labelling.plotting import prepare_data, make_plot, make_interactive_plot, suggest_clusters,suggestion_chart

def write(embset):
    chart_container,options_container=streamlit.beta_columns(2)
    xi=options_container.checkbox('use xi method')
    epsilon=options_container.slider('epsilon',0.0,10.0,4.0,0.1)
    min_samples=options_container.slider('min_samples',1,100,50,1)
    min_cluster_size=options_container.slider('min_cluster_size',1,100,30,1)
    if xi:
        xi_value=options_container.slider('xi value',0.0,1.0,0.05,0.01)

    if xi:
        algo = OPTICS(cluster_method='xi',eps=epsilon,min_samples=min_samples,min_cluster_size=min_cluster_size,xi=xi_value)
    else:
        algo = OPTICS(cluster_method='dbscan',eps=epsilon,min_samples=min_samples,min_cluster_size=min_cluster_size)
    
    df = suggest_clusters(embset, algo)
    view_cluster=options_container.selectbox('Select cluster to view:',['all']+df.labels.unique().tolist())
    if view_cluster=='all':
        data=df.copy()
    else:
        data=df.copy()
        data.labels=data.labels.apply(lambda x:1 if x==view_cluster else 0)
    chart=suggestion_chart(data)
    chart_container.altair_chart(chart,use_container_width=True)