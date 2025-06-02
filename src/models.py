import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin

class Model(BaseEstimator, ClassifierMixin):
    """A simple model class that implements sklearn's BaseEstimator and ClassifierMixin."""

    def __init__(self, param1=1.0, param2=0.5):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y):
        """Fit the model to the training data."""
        # Placeholder for fitting logic
        return self

    def predict(self, X):
        """Predict using the fitted model."""
        # Placeholder for prediction logic
        return [0] * len(X)  # Dummy prediction

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        # Placeholder for scoring logic
        return 0.5