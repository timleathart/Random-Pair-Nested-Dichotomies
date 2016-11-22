from rpnd import NestedDichotomy
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

data = load_iris()

X, y = data.data, data.target

lr = LogisticRegression()
nd = NestedDichotomy(lr)

nd.fit(X, y)

