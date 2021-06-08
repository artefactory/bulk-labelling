from lib.custom_whatlies.transformers._pca import Pca

from lib.custom_whatlies.transformers._noise import Noise
from lib.custom_whatlies.transformers._addrandom import AddRandom
from lib.custom_whatlies.transformers._tsne import Tsne
from lib.custom_whatlies.transformers._normalizer import Normalizer
from lib.custom_whatlies.transformers._transformer import SklearnTransformer, Transformer
from lib.custom_whatlies.error import NotInstalled

try:
    from lib.custom_whatlies.transformers._umap import Umap
except ImportError:
    Umap = NotInstalled("Umap", "umap")


__all__ = [
    "SklearnTransformer",
    "Transformer",
    "Pca",
    "Umap",
    "Tsne",
    "Noise",
    "AddRandom",
    "Normalizer",
]
