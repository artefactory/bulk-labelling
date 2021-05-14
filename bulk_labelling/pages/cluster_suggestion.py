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

def write(embset):
    chart_container,options_container=streamlit.beta_columns(2)
    xi=options_container.checkbox('use xi method')
    epsilon=options_container.slider('epsilon',0,100,1,1)
    min_samples=options_container.slider('min_samples',1,100,1,1)
    min_cluster_size=options_container.slider('min_cluster_size',1,100,1,1)
    if xi:
        xi_value=options_container.slider('xi value',0.0,1.0,0.5,0.01)

    if xi:
        algo = OPTICS(cluster_method='xi',eps=epsilon,min_samples=min_samples,min_cluster_size=min_cluster_size,xi=xi_value)
    else:
        algo = OPTICS(cluster_method='dbscan',eps=epsilon,min_samples=min_samples,min_cluster_size=min_cluster_size)
    
    chart, df = suggest_clusters(embset, algo)
    chart_container.altair_chart(chart,use_container_width=True)