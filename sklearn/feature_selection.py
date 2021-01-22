"""
Feature selection algorithms.

@author: nam
"""
import pandas as pd
from BorutaShap import BorutaShap

from sklearn.ensemble import RandomForestClassifier as RF


class PipeBorutaSHAP:
    """
    BorutaSHAP feature selector for pipelines.

    Create a BorutaSHAP instance that is compatible with
    scikit-learn's estimator API and can be used in sklearn and
    imblearn's pipelines.

    This is essentially a wrapper for
    [BorutaSHAP](https://github.com/Ekeany/Boruta-Shap). See
    documentation therein for additional details.
    """

    def __init__(
        self,
        column_names,
        model=RF(
            n_estimators=100,
            criterion="entropy",
            random_state=0,
            class_weight="balanced",
        ),
        classification=True,
        percentile=100,
        pvalue=0.05,
    ):
        """Instantiate the class."""
        self.set_params(
            **{
                "column_names": column_names,
                "model": model,
                "classification": classification,
                "percentile": percentile,
                "pvalue": pvalue,
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
            "column_names": self.column_names,
            "model": self.model,
            "classification": self.classification,
            "percentile": self.percentile,
            "pvalue": self.pvalue,
        }

    def fit(self, X, y):
        """Fit BorutaSHAP to data."""
        # Convert X and y to pandas.DataFrame and series
        self.__boruta_ = BorutaShap(
            model=self.model,
            importance_measure="shap",
            classification=self.classification,
            percentile=self.percentile,
            pvalue=self.pvalue,
        )

        assert X.shape[1] == len(
            self.column_names
        ), "X is not compatible \
        with column names provided."
        # BorutaSHAP is expensive so try to keep these to reasonable values.
        # If used in kfold CV the cost goes up very quickly.
        self.__boruta_.fit(
            X=pd.DataFrame(data=X, columns=self.column_names),
            y=pd.Series(data=y),
            n_trials=20,
            sample=False,
            train_or_test="test",  # Does internal 70:30 train/test
            normalize=True,
            verbose=False,
            random_state=0,
        )

        return self

    def transform(self, X):
        """Select the columns that were deemed important."""
        # Could reorder X relative to original input?
        return pd.DataFrame(data=X, columns=self.column_names)[
            self.accepted
        ].values

    @property
    def accepted(self):
        """Get the columns that are important."""
        return self.__boruta_.accepted

    @property
    def rejected(self):
        """Get the columns that are not important."""
        return self.__boruta_.rejected
