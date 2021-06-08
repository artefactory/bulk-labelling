from typing import Union, Optional, Sequence, Callable
from copy import deepcopy

import numpy as np
import scipy.spatial.distance as scipy_distance
from sklearn.metrics import pairwise_distances



class Embedding:
    """
    This object represents a word embedding. It contains a vector and a name.

    Arguments:
        name: the name of this embedding, includes operations
        vector: the numerical representation of the embedding
        orig: original name of embedding, is left alone

    Usage:

    ```python
    from lib.custom_whatlies.embedding import Embedding

    foo = Embedding("foo", [0.1, 0.3])
    bar = Embedding("bar", [0.7, 0.2])

    foo | bar
    foo - bar + bar
    ```
    """

    def __init__(self, name, vector, orig=None):
        self.orig = name if not orig else orig
        self.name = name
        self.vector = np.array(vector)

    def add_property(self, name, func):
        result = self.copy()
        setattr(result, name, func(result))
        return result

    @property
    def ndim(self):
        """
        Return the dimension of embedding vector.
        """
        return self.vector.shape[0]

    def copy(self):
        """
        Returns a deepcopy of the embdding.
        """
        return deepcopy(self)

    def __add__(self, other) -> "Embedding":
        """
        Add two embeddings together.

        Usage:

        ```python
        from lib.custom_whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])

        foo + bar
        ```
        """
        copied = deepcopy(self)
        copied.name = f"({self.name} + {other.name})"
        copied.vector = self.vector + other.vector
        return copied

    def __sub__(self, other):
        """
        Subtract two embeddings.

        Usage:

        ```python
        from lib.custom_whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])

        foo - bar
        ```
        """
        copied = deepcopy(self)
        copied.name = f"({self.name} - {other.name})"
        copied.vector = self.vector - other.vector
        return copied

    def __neg__(self):
        """
        Negate an embedding.

        Usage:

        ```python
        from lib.custom_whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])

        assert (- foo).vector == - foo.vector
        ```
        """
        copied = deepcopy(self)
        copied.name = f"(-{self.name})"
        copied.vector = -self.vector
        return copied

    def __gt__(self, other):
        """
        Measures the size of one embedding to another one.

        Usage:

        ```python
        from lib.custom_whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])

        foo > bar
        ```
        """
        return (self.vector.dot(other.vector)) / (other.vector.dot(other.vector))

    def __rshift__(self, other):
        """
        Maps an embedding unto another one.

        Usage:

        ```python
        from lib.custom_whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])

        foo >> bar
        ```
        """
        copied = deepcopy(self)
        new_vec = (
            (self.vector.dot(other.vector))
            / (other.vector.dot(other.vector))
            * other.vector
        )
        copied.name = f"({self.name} >> {other.name})"
        copied.vector = new_vec
        return copied

    def __or__(self, other):
        """
        Makes one embedding orthogonal to the other one.

        Usage:

        ```python
        from lib.custom_whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])

        foo | bar
        ```
        """
        copied = deepcopy(self)
        copied.name = f"({self.name} | {other.name})"
        copied.vector = self.vector - (self >> other).vector
        return copied

    def __repr__(self):
        return f"Emb[{self.name}]"

    def __str__(self):
        return self.name

    @property
    def norm(self):
        """Gives the norm of the vector of the embedding"""
        return np.linalg.norm(self.vector)

    def distance(self, other, metric: str = "cosine"):
        """
        Calculates the vector distance between two embeddings.

        Arguments:
            other: the other embedding you're comparing against
            metric: the distance metric to use, the list of valid options can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)

        **Usage**

        ```python
        from lib.custom_whatlies.embedding import Embedding

        foo = Embedding("foo", [1.0, 0.0])
        bar = Embedding("bar", [0.0, 0.5])

        foo.distance(bar)
        foo.distance(bar, metric="euclidean")
        foo.distance(bar, metric="cosine")
        ```
        """
        return pairwise_distances([self.vector], [other.vector], metric=metric)[0][0]
