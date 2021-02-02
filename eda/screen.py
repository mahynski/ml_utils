"""
Screening tools for features of data.

@author: nam
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    * This is a supervised method.

    * Using too many bins makes individual measurements all start to
    look unique and therefore 2 distributions appear to have a large
    JS divergence.  Be sure to try using a different number of bins
    to check your results qualitatively.  This also means outliers
    can be very problematic because they cause the the (max-min)
    range to be amplified artificially, which might actually make
    divergences look small because the bins are now too coarse.

    Example
    -------
    >>> screen = JSScreen(n=2, feature_names=X.columns)
    >>> screen.fit(X, y)
    >>> screen.visualize(plt.figure(figsize=(20,20)).gca())
    """

    def __init__(self, n=None, feature_names=None, js_bins=25):
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
    def merge(names, clause="AND", split=False):
        """Naming convention for merging classes."""
        if not clause.startswith(" "):
            clause = " " + clause
        if not clause.endswith(" "):
            clause = clause + " "
        if not split:
            # Merge together
            return clause.join(names)
        else:
            # Split apart
            return names.split(clause)

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

    def visualize_grid(self, ax=None):
        """Visualize the results with a heatmap."""
        if ax is None:
            ax = plt.figure().gca()

        ax = sns.heatmap(
            self.__grid_,
            ax=ax,
            annot=True,
            xticklabels=self.__column_labels_,
            yticklabels=self.__row_labels_,
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(r"$\nabla \cdot JS$")

    def visualize_classes(self, method="max", ax=None):
        """Visualize the classes by summarizing over the features."""
        if ax is None:
            ax = plt.figure().gca()

        if method == "mean":
            best = sorted(
                zip(
                    self.__column_labels_,
                    np.mean(self.__grid_, axis=0),
                    np.std(self.__grid_, axis=0),
                ),
                key=lambda x: x[1],
                reverse=True,
            )
        elif method == "max":
            best = sorted(
                zip(
                    self.__column_labels_,
                    np.max(self.__grid_, axis=0),
                    np.std(self.__grid_, axis=0),
                ),
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            raise ValueError("Unrecognized method")

        ax.bar(
            x=[x[0] for x in best],
            height=[x[1] for x in best],
            yerr=[x[2] for x in best],
        )
        ax.set_xticklabels([x[0] for x in best], rotation=90)
        ax.set_title("Feature {} +/- 1 ".format(method) + r"$\sigma$")
        ax.set_ylabel(r"$\nabla \cdot JS$")

        return best

    def visualize_max(self, k=None, bins=25):
        """
        Visualize the distribution of the max feature for classes.
        
        This will actually provide a visualization for all the top k
        macroclasses, so this is usually best when n=1 so only 
        individual atomic classes are visualized.

        Example
        -------
        >>> screen = JSScreen(n=1, feature_names=X.columns, js_bins=25)
        >>> screen.fit(X, y)
        >>> screen.visualize_max()
        """
        best = self.visualize_classes(method="max", ax=None)
        if k is None:
            k = len(best)

        top_feature = list(
            zip(
                self.__column_labels_,
                self.__row_labels_[np.argmax(self.__grid_, axis=0)],
            )
        )

        best_dict = {a: b for a, b, c in best}
        feat_dict = dict(top_feature)
        for class_, _ in sorted(
            {a: b for a, b, c in best}.items(), key=lambda x: x[1], reverse=True
        )[:k]:
            plt.figure()
            X_binary = pd.DataFrame(
                data=self.__X_, columns=self.feature_names
            )
            X_binary["class"] = self.__y_
            X_binary["class"][self.__y_ != class_] = "OTHER"
            ax = sns.histplot(
                hue="class",
                x=feat_dict[class_],
                data=X_binary,
                # multiple='stack',
                palette="Set1",
                stat="probability",
                bins=bins,
                common_norm=False,
            )
            ax.set_title(
                class_ + r"; $\nabla \cdot JS = {}$".format("%.3f"%best_dict[class_])
            )
            _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    @property
    def grid(self):
        """Get the grid of Jensen-Shannon divergences computed."""
        return self.__grid_.copy()

    def incremental(self, method="max"):
        """
        Find the changes due to the addition of a single class to a macroclass.

        For each macroclass, the difference of the method (max or mean) of the
        JS divergences is computed when one atomic class is added to the
        macroclass.  Postive values imply the new set is higher than before the
        class has been added; negative means adding the new class decreases the
        JS divergence (max or mean).

        The point at which a large positive occurs, suggests that newly formed
        macroclass represents a cluster that is separate from the other classes
        that is now complete.  A large negative change indicates that you have
        jsut added a class that overlaps the classes the in the macroclass; the
        larger the change, the more significant the overlap (i.e. maybe more of
        the constituent classes.

        Parameters
        ----------
        method : str
            Use the 'max' or the 'mean' of the JS divergences as the metric.

        Returns
        -------
        incremental : list([tuple(macroclass, addition), {'delta':change,
        'final':JS, 'individuals':{class:JS}}])
            The change that results from merging "addition" with macroclass,
            sorted from highest to lowest.  Note that this is a signed change,
            so you may wish to consider the magnitude instead.  The 'final' JS
            is the JS divergence of the new macroclass = macroclass + addition.
            The JS divergence for all the atomic classes is given in
            'individuals' for comparison.
        """
        if method == "max":
            function = np.max
        elif method == "mean":
            function = np.mean

        d = {}
        k = {}
        for j, combination in enumerate(self.__column_labels_):
            k[j] = set(self.merge(combination, split=True))
            d[j] = function(self.__grid_[:, j])

        def find(set_):
            for j, v in k.items():
                if v == set_:
                    return j
            raise ValueError("Could not find the set in question.")

        # Find which single additions resulted in the greatest "jumps"
        incremental = {}
        for j in d.keys():
            if len(k[j]) > 1:
                for x in k[j]:
                    idx = find(k[j].difference({x}))
                    delta = d[j] - d[idx]  # Find the value if x was removed
                    incremental[(self.__column_labels_[idx], x)] = {
                        "delta": delta,
                        "final": d[j],
                        "individuals": {c: d[find({c})] for c in k[j]},
                    }

        return sorted(
            incremental.items(), key=lambda x: x[1]["delta"], reverse=True
        )
