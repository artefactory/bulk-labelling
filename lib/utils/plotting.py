import streamlit
import pathlib
from lib.utils.embedding import get_embeddingset, get_language_array
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.transform import factor_cmap
import os
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import time
import logging
from bokeh.palettes import viridis,all_palettes, turbo
import pandas as pd





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
                  plot_width=370, plot_height=700, tooltips=TOOLTIPS,output_backend="webgl")

    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.outline_line_color = None
    plot.toolbar.logo = None









    plot.axis.visible = False

    if cluster:
        CLUSTERS = embset.cluster.unique().tolist()
        plot.circle("d1", "d2", source=cds_lasso,
                    color=factor_cmap('cluster', 'Category20_20',CLUSTERS), size=6, fill_alpha=0.7)
    else:
        LABELS = embset.labels.unique().tolist()
        # plot.circle("d1", "d2", source=cds_lasso,
        #              color='#3341F6', size=6, fill_alpha=0.7)
        plot.circle("d1", "d2", source=cds_lasso,
                    color=factor_cmap('labels', ('#3341F6', '#a6d2ff'), LABELS), size=6, fill_alpha=0.7)




    return plot




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
                          colormap='Blues',width = 1500, height = 1000).generate(bigtext.lower())
    
    return wordcloud



