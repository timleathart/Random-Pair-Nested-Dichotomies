from nested_dichotomy import NestedDichotomy
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import cross_val_predict

data = load_digits()

X, y = data.data, data.target

lr = LogisticRegression()
nd = NestedDichotomy(lr, subset_selection_method='random_pair')
bag = BaggingClassifier(nd)

preds = cross_val_predict(bag, X, y, cv=10)

print "RPND:", float((preds == y).sum()) / y.shape[0]

nd = NestedDichotomy(lr, subset_selection_method='random')
bag = BaggingClassifier(nd)

preds = cross_val_predict(bag, X, y, cv=10)

print "Random:", float((preds == y).sum()) / y.shape[0]
