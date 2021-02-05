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
from sklearn.preprocessing import LabelEncoder

from ml_utils.sklearn_ext.feature_selection import JensenShannonDivergence


class JSBinary:
    """
    Look at pairwise "separability" accodring to the JensenShannonDivergence.

    For a classification problem, look at the maximum JSD that can exists
    across all features between pairs of classes.  This creates a binary
    comparison between individual classes instead of a OvA comparison done in
    JSScreen.
    """

    def __init__(self, js_bins=25):
        """
        Instantiate the class.

        Parameters
        ----------
        js_bins : int
            Number of bins to use when computing the Jensen-Shannon
            divergence.
        """
        self.set_params(
            **{
                "js_bins": js_bins,
            }
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
            "js_bins": self.js_bins,
        }

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
        js = JensenShannonDivergence(
            **{
                "per_class": True,  # Sorts by max automatically
                "feature_names": None,  # Index
                "bins": self.js_bins,
            }
        )

        self.__enc_ = LabelEncoder()
        self.__enc_.fit(y)
        self.__matrix_ = np.zeros(
            (len(self.__enc_.classes_), len(self.__enc_.classes_))
        )
        self.__top_feature_ = np.empty(
            (len(self.__enc_.classes_), len(self.__enc_.classes_)), dtype=object
        )
        for pairs in itertools.combinations(np.unique(y), r=2):
            # 2. Compute (max) JS divergence
            mask = (y == pairs[0]) | (y == pairs[1])

            # Binary so divergences are the same, just take the first
            div = js.fit(X[mask], y[mask]).divergence
            x = div[pairs[0]][0][1][pairs[0]]
            feature = div[pairs[0]][0][0]
            assert div[pairs[1]][0][1][pairs[1]] == x

            i, j = self.__enc_.transform(pairs)
            self.__matrix_[i][j] = x
            self.__matrix_[j][i] = x
            self.__top_feature_[i][j] = feature
            self.__top_feature_[j][i] = feature

        return self

    @property
    def matrix(self):
        """Return the matrix of maximum JS divergence values."""
        return self.__matrix_.copy()

    def top_features(self, feature_names=None):
        """
        Return which feature was responsible for the max JS divergence.

        Parameters
        ----------
        feature_names : array-like
            List of feature names. Results are internally stored as
            indices so if this is provided, converts indices to names
            based on this array; otherwise a matrix of indices is
            returned.

        Example
        -------
        >>> jsb.top_features(feature_names=X.columns)
        """
        if feature_names is None:
            return self.__top_feature_.copy()
        else:
            names = np.empty_like(self.__top_feature_)
            for i in range(names.shape[0]):
                for j in range(names.shape[1]):
                    if i != j:
                        names[i, j] = feature_names[self.__top_feature_[i, j]]
                    else:
                        names[i, j] = "NONE"
            return names

    def visualize(self, ax=None):
        """Visualize the results with a heatmap."""
        if ax is None:
            ax = plt.figure().gca()

        ax = sns.heatmap(
            self.matrix,
            ax=ax,
            annot=True,
            xticklabels=self.__enc_.classes_,
            yticklabels=self.__enc_.classes_,
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(r"Maximum Pairwise $\nabla \cdot JS$")

        return ax


class JSScreen:
    """
    Use Jensen-Shannon divergences to screen for interesting features.

    For a classification problem, this uses JS divergences to
    combine classes in all possible ways to form "macroclasses."
    The JS divergence is then computed for all features when
    one class is the macroclass and all others are combined to
    form the opposing class (OvA method).

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
    * See sklearn_ext.feature_selection.JensenShannonDivergence for
    more discussion on the potential importance/impact of class
    imbalance with respect to bin size.

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
        n : int or None
            Maximum macroclass size; will return all combinations
            up to the point of containing n atomic classes.  In
            None, goes from 1 to len(atomic_classes).
        feature_names : list(str)
            Names of features (columns of X) in order.
        js_bins : int
            Number of bins to use when computing the Jensen-Shannon
            divergence.
        """
        self.set_params(
            **{
                "feature_names": np.array(feature_names, dtype=object),
                "n": n,
                "js_bins": js_bins,
            }
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
            "n": self.n,
            "feature_names": self.feature_names,
            "js_bins": self.js_bins,
        }

    @staticmethod
    def macroclasses(atomic_classes, n):
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

    @staticmethod
    def transform(y, macroclass, naming=None):
        """
        Transform classes into a macroclass.

        Parameters
        ----------
        y : array-like
            List of ground-truth classes.
        macroclass : tuple
            Tuple of classes that belong to the macroclass being created.
        naming : callable
            Function to name combinations of atomic classes; None defaults
            to the JSScreen.merge() method.

        Returns
        -------
        macro : array-like
            Classes after merging atomic ones into the macroclass.
        """
        y_macro = np.array(y)
        mask = np.array([x in macroclass for x in y_macro])
        namer = JSScreen.merge if naming is None else naming
        y_macro[mask] = namer(macroclass)

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
        mc = self.macroclasses(np.unique(y), n)
        transforms = {}
        for k, v in mc.items():
            transforms[k] = {}
            for i, macro in enumerate(v):
                transforms[k][self.merge(macro)] = self.transform(y, macro)

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
        self.__y_ = np.array(y, dtype=str)
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

    def visualize_classes(self, method="max", ax=None, display=True):
        """Visualize the classes by summarizing over the features."""
        if display:
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

        if display:
            ax.bar(
                x=[x[0] for x in best],
                height=[x[1] for x in best],
                yerr=[x[2] for x in best],
            )
            plt.xticks([x[0] for x in best], rotation=90)
            ax.set_title("Feature {} +/- 1 ".format(method) + r"$\sigma$")
            ax.set_ylabel(r"$\nabla \cdot JS$")

        return best

    def visualize_max(self, top=None, bins=25, ax=None):
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
        best = self.visualize_classes(method="max", ax=None, display=False)
        if top is None:
            top = len(best)

        top_feature = list(
            zip(
                self.__column_labels_,
                self.__row_labels_[np.argmax(self.__grid_, axis=0)],
            )
        )

        if ax is None:
            fig, axes = plt.subplots(nrows=top, ncols=1)
        else:
            axes = ax

        best_dict = {a: b for a, b, c in best}
        feat_dict = dict(top_feature)
        for ax_, (class_, _) in list(
            zip(
                axes.ravel(),
                sorted(best_dict.items(), key=lambda x: x[1], reverse=True),
            )
        )[:top]:
            X_binary = pd.DataFrame(data=self.__X_, columns=self.feature_names)
            y_ = self.__y_.copy()
            for c in self.merge(class_, split=True):
                y_[self.__y_ == c] = class_
            y_[y_ != class_] = "OTHER"
            X_binary["class"] = y_
            ax_ = sns.histplot(
                hue="class",
                x=feat_dict[class_],
                data=X_binary,
                # multiple='stack',
                palette="Set1",
                stat="probability",
                bins=bins,
                common_norm=False,
                ax=ax_,
            )
            ax_.set_title(
                class_
                + r"; $\nabla \cdot JS = {}$".format("%.3f" % best_dict[class_])
            )

    @property
    def grid(self):
        """Get the grid of Jensen-Shannon divergences computed."""
        return self.__grid_.copy()

    def interesting(self, threshold=0.7, method="max", min_delta=0.0):
        """
        Try to find the "interesting" macroclasses.

        In this example, we define "interesting merges" as those which cause a
        positive delta of at least `min_delta` and raise the JS divergence to
        above some `threshold` where it was initially below. Moreover, all the
        individual classes must have divergences less than the net of all of
        them less `min_delta` (i.e., merging is exclusively increasing the
        distinguishibility of the macroclass rather than one simply "bringing
        up the average").

        Example
        -------
        >>> interest = screen.interesting()
        >>> proposed_combinations = {}
        >>> performances = {}
        >>> idx = 0
        >>> for row in interest:
        ...     union = set(row[0][0].split(" AND ")).union({row[0][1]})
        ...     add = True
        ...     for k,v in proposed_combinations.items():
        ...         if v == union:
        ...             add = False
        ...             break
        ...     if add:
        ...         proposed_combinations[idx] = union
        ...         performances[idx] = row[1]['final']
        ...         idx += 1
        >>> proposed_combinations # Look at unique sets of interesting ones

        Returns
        -------
        incremental : list([tuple(macroclass, addition), {'delta':change,
        'final':JS, 'individuals':{class:JS}}])
            Merges that are considered "interesting."
        """
        interest = []
        for row in self.incremental(method=method):
            if (
                (row[1]["final"] > threshold)
                and (row[1]["final"] - row[1]["delta"]) < threshold
                and (row[1]["delta"] > min_delta)
                and np.all(
                    np.array(list(row[1]["individuals"].values()))
                    < row[1]["final"] - min_delta
                )
            ):
                interest.append(row)
        return interest

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
