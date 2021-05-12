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
from bulk_labelling.embedding import get_embeddingset, get_language_array
from hulearn.experimental.interactive import InteractiveCharts, SingleInteractiveChart
from bokeh.plotting import figure
from bokeh.models import DataTable, TableColumn,ColumnDataSource, CustomJS



@streamlit.cache
def prepare_data(lang, transformer, textlist=None):

    encoding, texts = get_language_array(lang, textlist)
    embset = get_embeddingset(encoding, texts)
    result = embset.transform(transformer)

    return result


@streamlit.cache(allow_output_mutation=True)
def make_plot(lang, transformer, textlist=None):
    return prepare_data(lang, transformer, textlist).plot_interactive(annot=False,x_label='',y_label='').properties(width=400, height=400,title='').configure_axisX(disable=True).configure_view(strokeOpacity =0).configure_mark(color='#3341F6')

@streamlit.cache(allow_output_mutation=True)
def make_interactive_plot(lang,transformer,textlist=None):
    df = prepare_data(lang,transformer,textlist).to_dataframe().reset_index()
    df.columns = ['text', 'd1', 'd2']
    df['label'] = ''
    plot = figure(tools="lasso_select,zoom_in,zoom_out",plot_width=400, plot_height=400)

    plot.axis.visible=False

    cds_lasso = ColumnDataSource(df)
    cds_lasso.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(source=cds_lasso),
            code="""
        document.dispatchEvent(
            new CustomEvent("LASSO_SELECT", {detail: {data: source.selected.indices}})
        )
        """
        )
    )
    plot.circle("d1", "d2", source=cds_lasso, color='#3341F6',size=6,fill_alpha=0.7)
    
    return plot,df
