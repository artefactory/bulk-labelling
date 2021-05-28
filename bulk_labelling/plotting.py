from bokeh.models.annotations import Tooltip
import streamlit
from bulk_labelling.custom_whatlies.embedding import Embedding
from bulk_labelling.custom_whatlies.embeddingset import EmbeddingSet
import pathlib
from bulk_labelling.embedding import get_embeddingset, get_language_array, cluster
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.io import export, export_png
# from sklearn.cluster import OPTICS, DBSCAN, MeanShift, estimate_bandwidth
import altair as alt
import os


@streamlit.cache
def prepare_data(lang, transformer, textlist=None, uuid=None):

    encoding, texts = get_language_array(lang, textlist)
    embset = get_embeddingset(encoding, texts)
    result = embset.transform(transformer)

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

    chart = alt.Chart(embed).mark_circle().encode(
        x=alt.X('d1', axis=None),
        y=alt.Y('d2', axis=None),
        color=alt.Color('labels', legend=None),
        tooltip=['text'],
    ).interactive().properties(width=400, height=400, title='').configure_view(strokeOpacity=0)
    return chart


@streamlit.cache(allow_output_mutation=True)
def make_interactive_plot(embset, show_labeled):

    if show_labeled:
        cds_lasso = ColumnDataSource(embset)
    else:
        cds_lasso = ColumnDataSource(embset[embset.labels == 'None'])
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
                  plot_width=400, plot_height=400, tooltips=TOOLTIPS)

    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.outline_line_color = None

    plot.axis.visible = False

    plot.circle("d1", "d2", source=cds_lasso,
                color='#3341F6', size=6, fill_alpha=0.7)

    

    return plot


def clear_cache():
    [f.unlink() for f in pathlib.Path("data/plotting_data/cache").glob("*")
     if (f.is_file() and not os.path.basename(f).startswith('.git'))]
