from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier, MetaEstimatorMixin
from sklearn.multiclass import _ConstantPredictor
import numpy as np

class NestedDichotomy(BaseEstimator, ClassifierMixin):
    
    def __init__(self, estimator, subset_selection_method="random", parent=None):
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
            print "Creating leaf node: ", y[0]
            
            self.estimator = _ConstantPredictor().fit(X, labels)
            return
        
        # Create split node if more than one class present
        pos_labels, neg_labels = None, None
        
        if self.subset_selection_method == "random":
            pos_labels, neg_labels = self._random_selection(labels)
        elif self.subset_selection_method == "class_balanced":
            pos_labels, neg_label = self._class_balanced(labels)

        # Generate new labels
        new_y = np.array(map(lambda y_val: 1 if y_val in pos_labels else 0, y))

        # Train base estimator to new labels
        self.estimator = clone(self.estimator)
        self.estimator.fit(X, new_y)

        # Recurse on left and right subtrees
        self.left = NestedDichotomy(self.estimator, subset_selection_method=self.subset_selection_method, parent=self)
        self.right = NestedDichotomy(self.estimator, subset_selection_method=self.subset_selection_method, parent=self)

        pos_X, pos_y = X[new_y == 1], y[new_y == 1]
        neg_X, neg_y = X[new_y == 0], y[new_y == 0]

        self.left.fit(pos_X, pos_y)
        self.right.fit(neg_X, neg_y)

    def predict(self, X):
        preds = self.estimator.predict(X)
        
        if isinstance(self.estimator, _ConstantPredictor):
            return preds
        else:
            left_X = X[preds==1]
            right_X = X[preds==0]
                        
            left_pred = self.left.predict(left_X)
            right_pred = self.right.predict(right_X)

            ret = np.zeros(X.shape[0])
            ret[preds==1] = left_pred
            ret[preds==0] = right_pred

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
        return labels.split














