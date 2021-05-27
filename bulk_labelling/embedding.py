
import streamlit

from bulk_labelling.custom_whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage, SpacyLanguage
from bulk_labelling.custom_whatlies.language import TFHubLanguage
from bulk_labelling.custom_whatlies.embedding import Embedding
from bulk_labelling.custom_whatlies.embeddingset import EmbeddingSet
from bulk_labelling.custom_whatlies.transformers import Pca, Umap, Tsne
from sentence_transformers import SentenceTransformer
from preshed.maps import PreshMap
from cymem.cymem import Pool
import json


@streamlit.cache
def get_embedding(vec, text):
    return Embedding(text, vec)


@streamlit.cache
def get_embeddingset(veclist, textlist):
    return EmbeddingSet(*[get_embedding(veclist[q], textlist[q]) for q in range(len(textlist))])


@streamlit.cache(allow_output_mutation=True)
def get_language_array(lang, textlist=None, uuid=None):
    if isinstance(lang, EmbeddingSet):
        return lang.to_names_X()[1], lang.to_names_X()[0]
    if isinstance(lang, SentenceTransformer):
        encoding = lang.encode(textlist)
        return encoding, textlist
    else:
        return lang[textlist].to_names_X()[1], lang[textlist].to_names_X()[0]


def cluster(algo, embset):
    transformed = embset[['d1', 'd2']].values

    clustering = algo.fit(transformed)

    labels = clustering.labels_
    embed = embset.copy()
    embed['labels'] = labels
    embed['labels'] = embed['labels'].astype(str)
    return embed
