"""
Screening tools for features of data.

@author: nam
"""

import itertools

import numpy as np
import seaborn as sns
import tqdm

from ml_utils.sklearn_ext.feature_selection import JensenShannonDivergence


class JSScreen:
    """
    Use Jensen-Shannon divergences to screen for interesting features.

    For a classification problem, this uses JS divergences to
    combine classes in all possible ways to form "macroclasses."
    The JS divergence is then computed for all features when
    one class is the macroclass and all others are combined to
    form the opposing class.

    This allows one to see if sets of classes can be separated from
    the rest of the "pack" according to certain features.  If
    so, this suggests that setting thresholds, or using trees,
    with those features could be an intuitive way to perform
    classification.

    Notes
    -----
    This is a supervised method.

    Example
    -------
    >>> screen = JSScreen(n=2, feature_names=X.columns)
    >>> screen.fit(X, y)
    >>> screen.visualize(plt.figure(figsize=(20,20)).gca())
    """

    def __init__(self, n=None, feature_names=None, js_bins=100):
        """
        Instantiate the class.

        Parameters
        ----------
        feature_names : list(str)
            Names of features (columns of X) in order.
        n : int or None
            Maximum macroclass size; will return all combinations
            up to the point of containing n atomic classes.  In
            None, goes from 1 to len(atomic_classes).
        js_bins : int
            Number of bins to use when computing the Jensen-Shannon
            divergence.
        """
        self.set_params(
            **{"feature_names": feature_names, "n": n, "js_bins": js_bins}
        )
        return

    def set_params(self, **parameters):
        """Set parameters; for consistency with sklearn's estimator API."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        """Get parameters; for consistency with sklearn's estimator API."""
        return {
            "feature_names": self.feature_names,
            "n": self.n,
            "js_bins": self.js_bins,
        }

    def macroclasses_(self, atomic_classes, n):
        """
        Create macroclasses from individual, atomic ones.

        Paremeters
        ----------
        atomic_classes : array-like
            List of classes, can strings or integers, for example.
        n : int or None
            Maximum macroclass size; will return all combinations
            up to the point of containing n atomic classes.  In
            None, goes from 1 to len(atomic_classes).

        Returns
        -------
        list(tuple) : macro
            List of combinations of atomic classes in order of n,
            following Pascal's triangle.
        """
        if n is not None:
            assert n >= 1
        macro = {}
        for i in range(1, (len(atomic_classes) if n is None else n) + 1):
            macro[i] = [x for x in itertools.combinations(atomic_classes, i)]

        return macro

    def transform_(self, y, macroclass):
        """
        Transform classes into a macroclass.

        Parameters
        ----------
        y : array-like
            List of ground-truth classes.
        macroclass : tuple
            Tuple of classes that belong to the macroclass being created.

        Returns
        -------
        macro : array-like
            Classes after merging atomic ones into the macroclass.
        """
        y_macro = np.array(y)
        mask = np.array([x in macroclass for x in y_macro])
        y_macro[mask] = self.merge(macroclass)

        return y_macro

    @staticmethod
    def merge(names):
        """Naming convention for merging classes."""
        return " AND ".join(names)

    def all_sets_(self, y, n):
        """
        Get all transformations of y into sets of [1:n].

        Parameters
        ----------
        y : array-like
            List of ground-truth classes.
        n : int or None
            Maximum macroclass size; will return all combinations
            up to the point of containing n atomic classes.  In
            None, goes from 1 to len(atomic_classes).

        Returns
        -------
        transforms : dict(dict)
            Dictionary of {n:{macroclass:y}}.
        """
        mc = self.macroclasses_(np.unique(y), n)
        transforms = {}
        for k, v in mc.items():
            transforms[k] = {}
            for i, macro in enumerate(v):
                transforms[k][self.merge(macro)] = self.transform_(y, macro)

        return transforms

    def fit(self, X, y):
        """
        Fit the screen to data.

        Parameters
        ----------
        X : array-like
            Features (columns) and observations (rows).
        y : array-like
            Ground truth classes.
        """
        self.__X_ = np.array(X)
        self.__y_ = np.array(y)
        assert self.__X_.shape[0] == self.__y_.shape[0]
        self.__transforms_ = self.all_sets_(self.__y_, self.n)

        self.__js_ = JensenShannonDivergence(
            **{
                "per_class": False,
                "feature_names": None,  # Index
                "bins": self.js_bins,
            }
        )

        self.__row_labels_ = (
            np.arange(X.shape[1])
            if self.feature_names is None
            else self.feature_names
        )  # Features are rows
        if self.feature_names is not None:
            assert len(self.feature_names) == self.__X_.shape[1]
        self.__column_labels_ = []  # Columns are macro-classes

        grid = []
        for n in tqdm.tqdm(self.__transforms_.keys()):
            for combine in tqdm.tqdm(self.__transforms_[n].keys()):
                self.__column_labels_.append(combine)
                y_ = self.__transforms_[n][combine]
                self.__js_.fit(self.__X_, y_)
                grid.append(
                    [
                        x[1]
                        for x in sorted(
                            {
                                a[0]: a[1][combine]
                                for a in self.__js_.divergence
                            }.items(),
                            key=lambda x: x[0],
                        )
                    ]
                )
        self.__grid_ = np.array(grid).T

        return self

    def visualize(self, ax=None):
        """Visualize the results with a heatmap."""
        ax = sns.heatmap(
            self.__grid_,
            ax=ax,
            annot=True,
            xticklabels=self.__column_labels_,
            yticklabels=self.__row_labels_,
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(r"$\nabla \cdot JS$")

    @property
    def grid(self):
        """Get the grid of Jensen-Shannon divergences computed."""
        return self.__grid_.copy()
