from bokeh import embed
from bokeh.models.annotations import Tooltip
import streamlit
from bulk_labelling.custom_whatlies.embedding import Embedding
from bulk_labelling.custom_whatlies.embeddingset import EmbeddingSet
import pathlib
from bulk_labelling.embedding import get_embeddingset, get_language_array, cluster
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.io import export, export_png
from bokeh.transform import factor_cmap
# from sklearn.cluster import OPTICS, DBSCAN, MeanShift, estimate_bandwidth
import altair as alt
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import time
import logging




@streamlit.cache
def prepare_data(lang, transformer, textlist=None, uuid=None):
    start = time.time()
    encoding, texts = get_language_array(lang, textlist)
    step1 = time.time()
    embset = get_embeddingset(encoding, texts)
    step2=time.time()
    result = embset.transform(transformer)
    end = time.time()
    logging.info(f'getting encodings : {step1-start}')
    logging.info(f'getting embeddingset : {step2-step1}')
    logging.info(f'getting transformation : {end-step2}')

    return result


@streamlit.cache(allow_output_mutation=True)
def make_plot(embed):
    embed_df = embed.reset_index()
    chart = alt.Chart(embed_df).mark_circle().encode(
        x=alt.X('d1', axis=None),
        y=alt.Y('d2', axis=None),
        tooltip=["text", "labels"]
    ).interactive().properties(width=400, height=400, title='').configure_view(strokeOpacity=0).configure_mark(color='#3341F6')
    return chart


def suggest_clusters(embset, algo):
    embed = cluster(algo, embset)
    return embed


def suggestion_chart(embed):

    cds_lasso = ColumnDataSource(embed)
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

    # chart = alt.Chart(embed).mark_circle().encode(
    #     x=alt.X('d1', axis=None),
    #     y=alt.Y('d2', axis=None),
    #     color=alt.Color('labels', legend=None),
    #     tooltip=['text'],
    # ).interactive().properties(width=400, height=400, title='').configure_view(strokeOpacity=0)
    # return chart

    TOOLTIPS = [
        ("text", "@text"),
    ]

    plot = figure(tools="lasso_select,zoom_in,zoom_out",
                  plot_width=370, plot_height=400, tooltips=TOOLTIPS)

    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.outline_line_color = None

    CLUSTERS = embed.cluster.unique().tolist()

    plot.axis.visible = False

    plot.circle("d1", "d2", source=cds_lasso,
                color=factor_cmap('cluster', 'Category10_3', CLUSTERS), size=6, fill_alpha=0.7)

    return plot


# @streamlit.cache(allow_output_mutation=True)
def make_interactive_plot(embset):

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

    # streamlit.write(cds_lasso.data)
    # streamlit.write(embset[embset.labels=='None'])

    plot = figure(tools="lasso_select,zoom_in,zoom_out",
                  plot_width=370, plot_height=400, tooltips=TOOLTIPS)

    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.outline_line_color = None

    plot.axis.visible = False

    plot.scatter("d1", "d2", source=cds_lasso,
                color='#3341F6', size=6, fill_alpha=0.7)

    return plot


def clear_cache():
    [f.unlink() for f in pathlib.Path("data/plotting_data/cache").glob("*")
     if (f.is_file() and not os.path.basename(f).startswith('.git'))]


def generate_wordcloud(textlist):
    bigtext = " ".join(textlist)
    stopwords_list = list(fr_stop) + list(en_stop) + \
        ['Très', 'Super', 'bien', 'bon']
    wordcloud = WordCloud(stopwords=stopwords_list, background_color="white",
                          colormap='Blues').generate(bigtext.lower())
    return wordcloud


def replace_labels(embedding_df, temp_embedding_df,label):
    embedding_df.loc[embedding_df.labelling_uuid.isin(temp_embedding_df[temp_embedding_df.labels!='None'].labelling_uuid),'labels']=label
    
    return embedding_df


