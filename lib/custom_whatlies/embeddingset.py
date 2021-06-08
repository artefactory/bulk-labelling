from copy import deepcopy
from functools import reduce
from collections import Counter
from typing import Union, Optional, Callable, Sequence, List

import numpy as np
import pandas as pd
import altair as alt
from sklearn.utils import deprecated
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import (
    paired_distances,
    cosine_similarity,
    cosine_distances,
    euclidean_distances,
)

from lib.custom_whatlies.embedding import Embedding


class EmbeddingSet:
    """
    This object represents a set of `Embedding`s. You can use the same operations
    as an `Embedding` but here we apply it to the entire set instead of a single
    `Embedding`.

    **Parameters**

    - **embeddings**: list of `Embedding`, or a single dictionary containing name:`Embedding` pairs
    - **name**: custom name of embeddingset

    Usage:

    ```python
    from whatlies.embedding import Embedding
    from whatlies.embeddingset import EmbeddingSet

    foo = Embedding("foo", [0.1, 0.3])
    bar = Embedding("bar", [0.7, 0.2])
    emb = EmbeddingSet(foo, bar)
    emb = EmbeddingSet({'foo': foo, 'bar': bar)
    ```
    """

    def __init__(self, *embeddings, name=None):
        if not name:
            name = "EmbSet"
        self.name = name
        if isinstance(embeddings[0], dict):
            # Assume it's a single dictionary.
            self.embeddings = embeddings[0]
        else:
            # Assume it's a list of `Embedding` instances.
            names = [t.name for t in embeddings]
            self.embeddings = {t.name: t for t in embeddings}

        # We cannot allow for different shapes because that will break many operations later.
        uniq_shapes = set(v.vector.shape for k, v in self.embeddings.items())
        if len(uniq_shapes) > 1:
            raise ValueError("Not all vectors have the same shape.")

    @property
    def ndim(self):
        """
        Return dimension of embedding vectors in embeddingset.
        """
        return next(iter(self.embeddings.values())).ndim

    def __contains__(self, item):
        """
        Checks if an item is in the embeddingset.

        Usage:

        ```python
        from whatlies.embedding import Embedding
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])
        buz = Embedding("buz", [0.1, 0.9])
        emb = EmbeddingSet(foo, bar)

        "foo" in emb # True
        "dinosaur" in emb # False
        ```
        """
        return item in self.embeddings.keys()

    def __iter__(self):
        """
        Iterate over all the embeddings in the embeddingset.

        Usage:

        ```python
        from whatlies.embedding import Embedding
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])
        buz = Embedding("buz", [0.1, 0.9])
        emb = EmbeddingSet(foo, bar)

        [e for e in emb]
        ```
        """
        return self.embeddings.values().__iter__()

    def __add__(self, other):
        """
        Adds an embedding to each element in the embeddingset.

        Usage:

        ```python
        from whatlies.embedding import Embedding
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])
        buz = Embedding("buz", [0.1, 0.9])
        emb = EmbeddingSet(foo, bar)

        (emb).plot(kind="arrow")
        (emb + buz).plot(kind="arrow")
        ```
        """
        new_embeddings = {k: emb + other for k, emb in self.embeddings.items()}
        return EmbeddingSet(new_embeddings, name=f"({self.name} + {other.name})")

    def __sub__(self, other):
        """
        Subtracts an embedding from each element in the embeddingset.

        Usage:

        ```python
        from whatlies.embedding import Embedding
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])
        buz = Embedding("buz", [0.1, 0.9])
        emb = EmbeddingSet(foo, bar)

        (emb).plot(kind="arrow")
        (emb - buz).plot(kind="arrow")
        ```
        """
        new_embeddings = {k: emb - other for k, emb in self.embeddings.items()}
        return EmbeddingSet(new_embeddings, name=f"({self.name} - {other.name})")

    def __or__(self, other):
        """
        Makes every element in the embeddingset othogonal to the passed embedding.

        Usage:

        ```python
        from whatlies.embedding import Embedding
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])
        buz = Embedding("buz", [0.1, 0.9])
        emb = EmbeddingSet(foo, bar)

        (emb).plot(kind="arrow")
        (emb | buz).plot(kind="arrow")
        ```
        """
        new_embeddings = {k: emb | other for k, emb in self.embeddings.items()}
        return EmbeddingSet(new_embeddings, name=f"({self.name} | {other.name})")

    def __rshift__(self, other):
        """
        Maps every embedding in the embedding set unto the passed embedding.

        Usage:

        ```python
        from whatlies.embedding import Embedding
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])
        buz = Embedding("buz", [0.1, 0.9])
        emb = EmbeddingSet(foo, bar)

        (emb).plot(kind="arrow")
        (emb >> buz).plot(kind="arrow")
        ```
        """
        new_embeddings = {k: emb >> other for k, emb in self.embeddings.items()}
        return EmbeddingSet(new_embeddings, name=f"({self.name} >> {other.name})")

    def compare_against(
        self, other: Union[str, Embedding], mapping: Optional[Callable] = None
    ) -> List:
        """
        Compare (or map) the embeddigns in the embeddingset to a given embedding, optionally using
        a custom mapping function.

        Arguments:
            other: an `Embedding` instance, or name of an existing embedding; it is used for
                comparison with each embedding in the embeddingset.
            mapping: an optional callable used for for comparison that takes two 1D vector arrays as
                input; if not given, the normalized scalar projection (i.e. `>` operator) is used.
        """
        if isinstance(other, str):
            other = self[other]
        if mapping is None:
            return [v > other for v in self.embeddings.values()]
        elif callable(mapping):
            return [mapping(v.vector, other.vector) for v in self.embeddings.values()]
        else:
            raise ValueError(f"Unrecognized mapping value/type, got: {mapping}")

    def pipe(self, func, *args, **kwargs):
        """
        Applies a function to the embedding set. Useful for method chaining and
        chunks of code that repeat.

        Arguments:
             func: callable that accepts an `EmbeddingSet` set as its first argument
             args: arguments to also pass to the function
             kwargs: keyword arguments to also pass to the function

        ```python
        from whatlies.language import SpacyLanguage, BytePairLanguage

        lang_sp = SpacyLanguage("en_core_web_sm")
        lang_bp = BytePairLanguage("en", dim=25, vs=1000)

        text = ["cat", "dog", "rat", "blue", "red", "yellow"]

        def make_plot(embset):
            return (embset
                    .plot_interactive("dog", "blue")
                    .properties(height=200, width=200))

        p1 = lang_sp[text].pipe(make_plot)
        p2 = lang_bp[text].pipe(make_plot)
        p1 | p2
        ```
        """
        return func(self, *args, **kwargs)

    def to_X(self, norm=False):
        """
        Takes every vector in each embedding and turns it into a scikit-learn compatible `X` matrix.

        Usage:

        ```python
        from whatlies.embedding import Embedding
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])
        buz = Embedding("buz", [0.1, 0.9])
        emb = EmbeddingSet(foo, bar, buz)

        X = emb.to_X()
        ```
        """
        X = np.array([i.vector for i in self.embeddings.values()])
        X = normalize(X) if norm else X
        return X

    def to_X_y(self, y_label):
        """
        Takes every vector in each embedding and turns it into a scikit-learn compatible `X` matrix.
        Also retreives an array with potential labels.

        Usage:

        ```python
        from whatlies.embedding import Embedding
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])
        buz = Embedding("buz", [0.1, 0.9])
        bla = Embedding("bla", [0.2, 0.8])

        emb1 = EmbeddingSet(foo, bar).add_property("label", lambda d: 'group-one')
        emb2 = EmbeddingSet(buz, bla).add_property("label", lambda d: 'group-two')
        emb = emb1.merge(emb2)

        X, y = emb.to_X_y(y_label='label')
        ```
        """
        X = self.to_X()
        y = np.array([getattr(e, y_label) for e in self.embeddings.values()])
        return X, y

    def to_names_X(self):
        """
        Get the list of names as well as an array of vectors of all embeddings.

        Usage:

        ```python
        from whatlies.embedding import Embedding
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])
        buz = Embedding("buz", [0.1, 0.9])
        emb = EmbeddingSet(foo, bar, buz)

        names, X = emb.to_names_X()
        ```
        """
        return list(self.embeddings.keys()), self.to_X()

    @classmethod
    def from_names_X(cls, names, X):
        """
        Constructs an `EmbeddingSet` instance from the given embedding names and vectors.

        Arguments:
            names: an iterable containing the names of embeddings
            X: an iterable of 1D vectors, or a 2D numpy array; it should have the same length as `names`

        Usage:

        ```python
        from whatlies.embeddingset import EmbeddingSet

        names = ["foo", "bar", "buz"]
        vecs = [
            [0.1, 0.3],
            [0.7, 0.2],
            [0.1, 0.9],
        ]

        emb = EmbeddingSet.from_names_X(names, vecs)
        ```
        """
        X = np.array(X)
        if len(X) != len(names):
            raise ValueError(
                f"The number of given names ({len(names)}) and vectors ({len(X)}) should be the same."
            )
        return cls({n: Embedding(n, v) for n, v in zip(names, X)})

    def transform(self, transformer):
        """
        Applies a transformation on the entire set.

        Usage:

        ```python
        from whatlies.embeddingset import EmbeddingSet
        from whatlies.transformers import Pca

        foo = Embedding("foo", [0.1, 0.3, 0.10])
        bar = Embedding("bar", [0.7, 0.2, 0.11])
        buz = Embedding("buz", [0.1, 0.9, 0.12])
        emb = EmbeddingSet(foo, bar, buz).transform(Pca(2))
        ```
        """
        return transformer(self)

    def __getitem__(self, thing):
        """
        Retreive a single embedding from the embeddingset.

        Usage:
        ```python
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3, 0.10])
        bar = Embedding("bar", [0.7, 0.2, 0.11])
        buz = Embedding("buz", [0.1, 0.9, 0.12])
        emb = EmbeddingSet(foo, bar, buz)

        emb["buz"]
        ```
        """
        if isinstance(thing, str):
            return self.embeddings[thing]
        new_embeddings = {t: self[t] for t in thing}
        names = ",".join(thing)
        return EmbeddingSet(new_embeddings, name=f"{self.name}.subset({names})")

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.embeddings.keys())

    def filter(self, func):
        """
        Filters the collection of embeddings based on a predicate function.

        Arguments:
             func: callable that accepts a single embedding and outputs a boolean

        ```python
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3, 0.10])
        bar = Embedding("bar", [0.7, 0.2, 0.11])
        buz = Embedding("buz", [0.1, 0.9, 0.12])
        xyz = Embedding("xyz", [0.1, 0.9, 0.12])
        emb = EmbeddingSet(foo, bar, buz, xyz)
        emb.filter(lambda e: "foo" not in e.name)
        ```
        """
        return EmbeddingSet({k: v for k, v in self.embeddings.items() if func(v)})

    def merge(self, other):
        """
        Concatenates two embeddingssets together

        Arguments:
            other: another embeddingset

        Usage:

        ```python
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3, 0.10])
        bar = Embedding("bar", [0.7, 0.2, 0.11])
        buz = Embedding("buz", [0.1, 0.9, 0.12])
        xyz = Embedding("xyz", [0.1, 0.9, 0.12])
        emb1 = EmbeddingSet(foo, bar)
        emb2 = EmbeddingSet(xyz, buz)

        both = emb1.merge(emb2)
        ```
        """
        return EmbeddingSet({**self.embeddings, **other.embeddings})

    def assign(self, **kwargs):
        """
        Adds properties to every embedding in the set based on the keyword arguments.

        This is very useful for plotting because a property can be used to assign colors. This method is very
        similar to `.add_property` but it might be more convenient when you want to assign multiple properties
        in one single statement.

        Arguments:
            kwargs: (name, func)-pairs that describe the name of the property as well as a value to assign.
                The value can be a single value, iterable or a function. The function expects an `Embedding` object as input.

        Usage:

        ```python
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3, 0.10])
        bar = Embedding("bar", [0.7, 0.2, 0.11])
        emb = EmbeddingSet(foo, bar)
        emb_with_property1 = emb.assign(dim0=lambda d: d.vector[0],
                                        dim1=lambda d: d.vector[1],
                                        dim2=lambda d: d.vector[2])

        emb_with_property2 = emb.assign(group=["foo_grp", "bar_grp"])

        emb_with_property3 = emb.assign(constant=1)
        ```
        """
        new_set = {}
        for idx, (k, e) in enumerate(self.embeddings.items()):
            new_emb = e
            for name, val in kwargs.items():
                if callable(val):
                    new_emb = new_emb.add_property(name, val)
                elif hasattr(val, "__iter__") and not isinstance(val, str):
                    # We want to support lists, tuples, numpy arrays but not strings
                    # those need to be handle as if they're literals.
                    if len(val) != len(self):
                        raise ValueError(
                            f"If you're passing an iterable to `.assign` then it must have the same length as the `EmbeddingSet`.\nGot: {len(val)}. Expected: {len(self)}."
                        )
                    new_emb = new_emb.add_property(name, lambda d: val[idx])
                else:
                    new_emb = new_emb.add_property(name, lambda d: val)
            new_set[k] = new_emb
        return EmbeddingSet(new_set)

    def add_property(self, name, func):
        """
        Adds a property to every embedding in the set. Very useful for plotting because
        a property can be used to assign colors.

        Arguments:
            name: name of the property to add
            func: function that receives an embedding and needs to output the property value

        Usage:

        ```python
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [0.1, 0.3, 0.10])
        bar = Embedding("bar", [0.7, 0.2, 0.11])
        emb = EmbeddingSet(foo, bar)
        emb_with_property = emb.add_property('example', lambda d: 'group-one')
        ```
        """
        return EmbeddingSet(
            {k: e.add_property(name, func) for k, e in self.embeddings.items()}
        )

    def average(self, name=None):
        """
        Takes the average over all the embedding vectors in the embeddingset. Turns it into
        a new `Embedding`.

        Arguments:
            name: manually specify the name of the average embedding

        Usage:

        ```python
        from whatlies.embeddingset import EmbeddingSet

        foo = Embedding("foo", [1.0, 0.0])
        bar = Embedding("bar", [0.0, 1.0])
        emb = EmbeddingSet(foo, bar)

        emb.average().vector                   # [0.5, 0,5]
        emb.average(name="the-average").vector # [0.5, 0.5]
        ```
        """
        name = f"{self.name}.average()" if not name else name
        x = self.to_X()
        return Embedding(name, np.mean(x, axis=0))

    def embset_similar(self, emb: Union[str, Embedding], n: int = 10, metric="cosine"):
        """
        Retreive an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] that are the most simmilar to the passed query.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            metric: metric to use to calculate distance, must be scipy or sklearn compatible

        Returns:
            An [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] containing the similar embeddings.
        """
        embs = [w[0] for w in self.score_similar(emb, n, metric)]
        return EmbeddingSet({w.name: w for w in embs})

    def score_similar(self, emb: Union[str, Embedding], n: int = 10, metric="cosine"):
        """
        Retreive a list of (Embedding, score) tuples that are the most similar to the passed query.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            metric: metric to use to calculate distance, must be scipy or sklearn compatible

        Returns:
            An list of ([Embedding][whatlies.embedding.Embedding], score) tuples.
        """
        if n > len(self):
            raise ValueError(
                f"You cannot retreive (n={n}) more items than exist in the Embeddingset (len={len(self)})"
            )

        if isinstance(emb, str):
            if emb not in self.embeddings.keys():
                raise ValueError(
                    f"Embedding for `{emb}` does not exist in this EmbeddingSet"
                )
            emb = self[emb]

        vec = emb.vector
        queries = [w for w in self.embeddings.keys()]
        vector_matrix = self.to_X()
        distances = pairwise_distances(vector_matrix, vec.reshape(1, -1), metric=metric)
        by_similarity = sorted(zip(queries, distances), key=lambda z: z[1])
        return [(self[q], float(d)) for q, d in by_similarity[:n]]

    def to_matrix(self):
        """
        Does exactly the same as `.to_X`. It takes the embedding vectors and turns it into a numpy array.
        """
        return self.to_X()

    def to_dataframe(self):
        """
        Turns the embeddingset into a pandas dataframe.
        """
        mat = self.to_matrix()
        return pd.DataFrame(mat, index=list(self.embeddings.keys()))

    def movement_df(self, other, metric="euclidean"):
        """
        Creates a dataframe that shows the movement from one embeddingset to another one.

        Arguments:
            other: the other embeddingset to compare against, will only keep the overlap
            metric: metric to use to calculate movement, must be scipy or sklearn compatible

        Usage:

        ```python
        from whatlies.language import SpacyLanguage

        lang = SpacyLanguage("en_core_web_sm")

        names = ['red', 'blue', 'green', 'yellow', 'cat', 'dog', 'mouse', 'rat', 'bike', 'car']
        emb = lang[names]
        emb_ort = lang[names] | lang['cat']
        emb.movement_df(emb_ort)
        ```
        """
        overlap = list(
            set(self.embeddings.keys()).intersection(set(other.embeddings.keys()))
        )
        mat1 = np.array([w.vector for w in self[overlap]])
        mat2 = np.array([w.vector for w in other[overlap]])
        return (
            pd.DataFrame(
                {"name": overlap, "movement": paired_distances(mat1, mat2, metric)}
            )
            .sort_values(["movement"], ascending=False)
            .reset_index()
        )

    def to_axis_df(self, x_axis, y_axis):
        if isinstance(x_axis, str):
            x_axis = self[x_axis]
        if isinstance(y_axis, str):
            y_axis = self[y_axis]
        return pd.DataFrame(
            {
                "x_axis": self.compare_against(x_axis),
                "y_axis": self.compare_against(y_axis),
                "name": [v.name for v in self.embeddings.values()],
                "original": [v.orig for v in self.embeddings.values()],
            }
        )

    def plot(
        self,
        kind: str = "arrow",
        x_axis: Union[int, str, Embedding] = 0,
        y_axis: Union[int, str, Embedding] = 1,
        axis_metric: Optional[Union[str, Callable, Sequence]] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        title: Optional[str] = None,
        color: str = None,
        show_ops: bool = False,
        annot: bool = True,
        axis_option: Optional[str] = None,
    ):
        """
        Makes (perhaps inferior) matplotlib plot. Consider using `plot_interactive` instead.

        Arguments:
            kind: what kind of plot to make, can be `scatter`, `arrow` or `text`
            x_axis: the x-axis to be used, must be given when dim > 2; if an integer, the corresponding
                dimension of embedding is used.
            y_axis: the y-axis to be used, must be given when dim > 2; if an integer, the corresponding
                dimension of embedding is used.
            axis_metric: the metric used to project each embedding on the axes; only used when the corresponding
                axis (i.e. `x_axis` or `y_axis`) is a string or an `Embedding` instance. It could be a string
                (`'cosine_similarity'`, `'cosine_distance'` or `'euclidean'`), or a callable that takes two vectors as input
                and returns a scalar value as output. To set different metrics for x- and y-axis, a list or a tuple of
                two elements could be given. By default (`None`), normalized scalar projection (i.e. `>` operator) is used.
            x_label: an optional label used for x-axis; if not given, it is set based on value of `x_axis`.
            y_label: an optional label used for y-axis; if not given, it is set based on value of `y_axis`.
            title: an optional title for the plot.
            color: the color of the dots
            show_ops: setting to also show the applied operations, only works for `text`
            annot: should the points be annotated
            axis_option: a string which is passed as `option` argument to `matplotlib.pyplot.axis` in order to control
                axis properties (e.g. using `'equal'` make circles shown circular in the plot). This might be useful
                for preserving geometric relationships (e.g. orthogonality) in the generated plot. See `matplotlib.pyplot.axis`
                [documentation](https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.axis.html#matplotlib-pyplot-axis)
                for possible values and their description.
        """
        if isinstance(x_axis, str):
            x_axis = self[x_axis]
        if isinstance(y_axis, str):
            y_axis = self[y_axis]

        if isinstance(axis_metric, (list, tuple)):
            x_axis_metric = axis_metric[0]
            y_axis_metric = axis_metric[1]
        else:
            x_axis_metric = axis_metric
            y_axis_metric = axis_metric

        embeddings = []
        for emb in self.embeddings.values():
            x_val, x_lab = emb._get_plot_axis_value_and_label(
                x_axis, x_axis_metric, dir="x"
            )
            y_val, y_lab = emb._get_plot_axis_value_and_label(
                y_axis, y_axis_metric, dir="y"
            )
            emb_plot = Embedding(name=emb.name, vector=[x_val, y_val], orig=emb.orig)
            embeddings.append(emb_plot)
        x_label = x_lab if x_label is None else x_label
        y_label = y_lab if y_label is None else y_label
        handle_2d_plot(
            embeddings,
            kind=kind,
            color=color,
            xlabel=x_label,
            ylabel=y_label,
            title=title,
            show_operations=show_ops,
            annot=annot,
            axis_option=axis_option,
        )
        return self

    