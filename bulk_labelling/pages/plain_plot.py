import streamlit
from bulk_labelling.custom_whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage, SpacyLanguage
from bulk_labelling.custom_whatlies.language import TFHubLanguage
from bulk_labelling.custom_whatlies.embedding import Embedding
from bulk_labelling.custom_whatlies.embeddingset import EmbeddingSet
from bulk_labelling.custom_whatlies.transformers import Pca, Umap, Tsne
from preshed.maps import PreshMap
from cymem.cymem import Pool


from bulk_labelling.load_config import load_languages, load_transformer, load_dataset, load_config
from bulk_labelling.embedding import get_embedding, get_embeddingset, get_language_array
from bulk_labelling.plotting import prepare_data, make_plot, make_interactive_plot, suggest_clusters

def write(embset,dataset):
    chart_container,options_container=streamlit.beta_columns(2)
    chart = make_plot(embset)
    chart_container.altair_chart(chart, use_container_width=True)
    options_container.info(f'looking at {len(dataset)} examples')