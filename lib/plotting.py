import streamlit
import pathlib
from lib.embedding import get_embeddingset, get_language_array
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.transform import factor_cmap
import os
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import time
import logging


def prepare_data(lang, transformer, textlist=None):
    """encodes and transforms a list of texts with the models chosen by the user

    Args:
        lang (huggingface transformer, tfhub language or whatlies language): model used to encode the texts
        transformer (whatlies transformer): dimension reduction transformer chosn by the user
        textlist (list, optional): list of texts to encode. Defaults to None.

    Returns:
        pd.DataFrame: dataframe containing the texts to encode as well as their n-D transformed encodings
    """
    start = time.time()
    encoding, texts = get_language_array(lang, textlist)
    step1 = time.time()
    embset = get_embeddingset(encoding, texts)
    step2 = time.time()
    # embset = embset.transform(transformer)
    end = time.time()
    logging.info(f'getting encodings : {step1-start}')
    logging.info(f'getting embeddingset : {step2-step1}')

    return embset



def make_interactive_plot(embset, cluster):
    """makes an interactive plot given a dataframe with the x and y data

    Args:
        embset (pd.Dataframe()): dataframe containing x,y data as well as encoded texts
        cluster (bool): should clusters be suggested

    Returns:
        bokeh.plotting.figure: a bokeh interactive plot with lasso-select update capabilities
    """    
    cds_lasso = ColumnDataSource(embset)
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
    TOOLTIPS = [
        ("text", "@text"),
    ]

    plot = figure(tools="lasso_select,zoom_in,zoom_out",
                  plot_width=370, plot_height=400, tooltips=TOOLTIPS,output_backend="webgl")

    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.outline_line_color = None

    plot.axis.visible = False

    if cluster:
        CLUSTERS = embset.cluster.unique().tolist()
        plot.circle("d1", "d2", source=cds_lasso,
                    color=factor_cmap('cluster', 'Category10_3', CLUSTERS), size=6, fill_alpha=0.7)
    else:
        plot.circle("d1", "d2", source=cds_lasso,
                     color='#3341F6', size=6, fill_alpha=0.7)


                     

    return plot


def clear_cache():
    """clears the handmade cache
    """
    [f.unlink() for f in pathlib.Path("data/plotting_data/cache").glob("*")
     if (f.is_file() and not os.path.basename(f).startswith('.git'))]


def generate_wordcloud(textlist):
    """generates a wordcloud from the a given text list

    Args:
        textlist (list): list of texts to generate a wordcloud from

    Returns:
        wordcloud.Wordcloud: wordcloud generated from the textlist
    """
    bigtext = " ".join(textlist)
    stopwords_list = list(fr_stop) + list(en_stop) + \
        ['Très', 'Super', 'bien', 'bon']
    wordcloud = WordCloud(stopwords=stopwords_list, background_color="white",
                          colormap='Blues').generate(bigtext.lower())
    return wordcloud


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
