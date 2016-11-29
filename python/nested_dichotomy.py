import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.base import clone, is_classifier
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import _ConstantPredictor

subset_selection_methods = [
    "random",
    "class_balanced",
    "random_pair"
]


class NestedDichotomy(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 estimator,
                 subset_selection_method="random",
                 parent=None):

        if subset_selection_method not in subset_selection_methods:
            raise ValueError(
                "%s is not a valid subset selection method" % (
                    subset_selection_method,))

        self.estimator = estimator
        self.subset_selection_method = subset_selection_method
        self.parent = parent

    def fit(self, X, y):
        labels = np.unique(y)

        # Set classes attribute
        if self.parent is None:
            self.classes_ = labels
        else:
            self.classes_ = self.parent.classes_

        # Create leaf node if only one class present
        if len(labels) == 1:
            self.estimator = _ConstantPredictor().fit(X, labels)
            return

        # Create split node if more than one class present
        pos_labels, neg_labels = None, None

        if len(labels) == 2:
            pos_labels, neg_labels = np.array_split(labels, 2)
        else:
            if self.subset_selection_method == "random":
                pos_labels, neg_labels = self._random_selection(labels)
            elif self.subset_selection_method == "class_balanced":
                pos_labels, neg_labels = self._class_balanced(labels)
            elif self.subset_selection_method == "random_pair":
                pos_labels, neg_labels = self._random_pair_selection(
                    labels, X, y)

        # Generate new labels
        new_y = np.array(map(lambda y_val: 1 if y_val in pos_labels else 0, y))

        # Train base estimator to new labels
        self.estimator = clone(self.estimator)
        self.estimator.fit(X, new_y)

        # Recurse on left and right subtrees
        self.left = NestedDichotomy(
            self.estimator,
            subset_selection_method=self.subset_selection_method,
            parent=self)

        self.right = NestedDichotomy(
            self.estimator,
            subset_selection_method=self.subset_selection_method,
            parent=self)

        pos_X, pos_y = X[new_y == 1], y[new_y == 1]
        neg_X, neg_y = X[new_y == 0], y[new_y == 0]

        self.left.fit(pos_X, pos_y)
        self.right.fit(neg_X, neg_y)

    def predict(self, X):
        preds = self.estimator.predict(X)

        if isinstance(self.estimator, _ConstantPredictor):
            return preds
        else:
            left_X = X[preds == 1]
            right_X = X[preds == 0]

            left_pred = self.left.predict(left_X)
            right_pred = self.right.predict(right_X)

            ret = np.zeros(X.shape[0])
            ret[preds == 1] = left_pred
            ret[preds == 0] = right_pred

            return ret

    def predict_proba(self, X):
        if isinstance(self.estimator, _ConstantPredictor):
            preds = self.estimator.predict(X)
            dist = np.eye(len(self.classes_))[preds]
            return dist
        else:
            left = self.left.predict_proba(X)
            right = self.right.predict_proba(X)

            factor = self.estimator.predict_proba(X)[:, 0].reshape((-1, 1))

            return factor * right + (1-factor) * left

    def _random_selection(self, labels):
        np.random.shuffle(labels)
        split_point = np.random.randint(1, high=len(labels))

        return np.split(labels, [split_point])

    def _class_balanced(self, labels):
        np.random.shuffle(labels)

        return np.array_split(labels, 2)

    def _random_pair_selection(self, labels, X, y):
        sorted_labels = np.sort(self.classes_)
        np.random.shuffle(labels)

        pos_labels = [labels[0]]
        neg_labels = [labels[1]]

        idx = np.logical_or(y == labels[0], y == labels[1])

        new_y = y[idx]

        # Generate new labels for random pair
        new_y = np.array(
            map(lambda y_val: 1 if y_val in pos_labels else 0, new_y))

        # Train base estimator to new labels
        self.estimator = clone(self.estimator)
        self.estimator.fit(X[idx], new_y)

        # TODO: test the remaining data on the estimator and make groups based on them
        preds = self.estimator.predict(X[~idx])

        cm = confusion_matrix(y[~idx], preds, labels=sorted_labels)

        for y_val in labels:
            if y_val in labels[:2]:
                continue
                
            if cm[y_val][0] < cm[y_val][1]:
                pos_labels += [y_val]
            else:
                neg_labels += [y_val]      
        
        return pos_labels, neg_labels







