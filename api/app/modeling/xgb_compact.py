"""XGBClassifier for labels with gaps. XGBoost wants classes 0..k-1, but a chain's cross-validation folds, or a band
with no training rows, can leave a target with classes {0, 2}; scikit-learn's classifiers accept that."""
import numpy as np
from xgboost import XGBClassifier


class CompactXGBClassifier(XGBClassifier):
    def fit(self, X, y, **kw):
        labels = np.unique(y)
        self.__dict__.pop("labels_", None)  # XGBoost reads classes_ during fit; it must be its own until then
        super().fit(X, np.searchsorted(labels, y), **kw)
        self.labels_ = labels
        return self

    @property
    def classes_(self):
        labels = self.__dict__.get("labels_")
        return labels if labels is not None else XGBClassifier.classes_.fget(self)

    def predict(self, X, **kw):
        return self.labels_[np.asarray(super().predict(X, **kw), dtype=int)]
