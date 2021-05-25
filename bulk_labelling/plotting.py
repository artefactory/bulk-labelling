import streamlit
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import datasets
# import pathlib
# from streamlit.uploaded_file_manager import UploadedFile
# import tensorflow as tf
# import tensorflow_hub
# from whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage, SpacyLanguage
# from whatlies.language import TFHubLanguage
from whatlies import Embedding, EmbeddingSet
import pathlib
# from whatlies.transformers import Pca, Umap, Tsne, Lda
# from sentence_transformers import SentenceTransformer
# from preshed.maps import PreshMap
# from cymem.cymem import Pool
from bulk_labelling.embedding import get_embeddingset, get_language_array, cluster
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
# from sklearn.cluster import OPTICS, DBSCAN, MeanShift, estimate_bandwidth
import altair as alt


@streamlit.cache
def prepare_data(lang, transformer, textlist=None,uuid=None):

    encoding, texts = get_language_array(lang, textlist)
    embset = get_embeddingset(encoding, texts)
    result = embset.transform(transformer)

    return result


@streamlit.cache(allow_output_mutation=True)
def make_plot(embed):
    embed_df=embed.reset_index()
    chart = alt.Chart(embed_df).mark_circle().encode(
        x=alt.X('d1', axis=None),
        y=alt.Y('d2', axis=None),
        tooltip=["text"]
    ).interactive().properties(width=400, height=400, title='').configure_view(strokeOpacity=0).configure_mark(color='#3341F6')
    return chart


def suggest_clusters(embset,algo):
    embed = cluster(algo, embset)
    return embed

def suggestion_chart(embed):
    
    chart = alt.Chart(embed).mark_circle().encode(
        x=alt.X('d1', axis=None),
        y=alt.Y('d2', axis=None),
        color=alt.Color('labels', legend=None),
        tooltip=['text'],
    ).interactive().properties(width=400, height=400, title='').configure_view(strokeOpacity=0)
    return chart


@streamlit.cache(allow_output_mutation=True)
def make_interactive_plot(embset):
    df = embset.copy().reset_index()
    plot = figure(tools="lasso_select,zoom_in,zoom_out",
                  plot_width=400, plot_height=400)

    plot.axis.visible = False

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
    plot.circle("d1", "d2", source=cds_lasso,
                color='#3341F6', size=6, fill_alpha=0.7)

    return plot, df

def clear_cache():
    [f.unlink() for f in pathlib.Path("data/plotting_data").glob("*") if f.is_file()]
