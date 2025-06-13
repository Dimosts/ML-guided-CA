from cpmpy.expressions.utils import is_any_list
from sklearn.base import ClassifierMixin, BaseEstimator
from collections import Counter


class PredictorTemplate(ClassifierMixin, BaseEstimator):
    """
    A template class for predictors, inheriting from scikit-learn's ClassifierMixin and BaseEstimator.
    """

    def __init__(self):
        """
        Initialize the predictor with the necessary parameters.
        """
        pass

    def fit(self, X, Y, **kwargs):
        """
        Fit the model using the provided training data.

        :param X: Set of constraint instances (featurized).
        :param Y: Set of labels for constraints.
        :param kwargs: Other arguments.
        :return: self
        """
        raise NotImplementedError

    def predict_proba(self, c):
        """
        Predict the probabilities for a given constraint.

        :param c: A constraint (featurized).
        :return: A tuple of probabilities (proba_false, proba_true).
        """
        raise NotImplementedError


class CountsPredictor(PredictorTemplate):
    """
    A predictor that uses counts of positive and negative instances to predict probabilities.
    Presented in:
    Dimos Tsouros, Senne Berden, and Tias Guns. "Guided Bottom-Up Interactive Constraint Acquisition." CP, 2023
    """

    def __init__(self):
        """
        Initialize the CountsPredictor.
        """
        super().__init__()
        self._theta = dict()
        self._counts_pos = dict()
        self._counts_neg = dict()

    def fit(self, X, Y, **kwargs):
        """
        Fit the model using the provided training data.

        :param X: List of constraint instances (featurized).
        :param Y: List of labels for constraints.
        :return: self
        """
        assert len(X[0]) == 1, "CountsPredictor works with only one feature."
        X = [x[0] for x in X]

        # Clear previous values
        self._theta = dict()
        self._counts_pos = dict()
        self._counts_neg = dict()

        # Count positive and negative instances directly
        pos_counter = Counter()
        neg_counter = Counter()
        for x, y in zip(X, Y):
            if y:
                pos_counter[x] += 1
            else:
                neg_counter[x] += 1

        # Get unique values from both counters
        unique_values = set(pos_counter.keys()) | set(neg_counter.keys())

        # Store counts and calculate theta
        for val in unique_values:
            self._counts_pos[val] = pos_counter[val]
            self._counts_neg[val] = neg_counter[val]
            self._theta[val] = ((self._counts_pos[val] + 0.25) / (self._counts_neg[val] + 0.5))

        return self

    def predict_proba(self, X):
        """
        Predict the probabilities for the given constraints.

        :param X: A list of featurized constraints.
        :return: List of probabilities for each constraint.
        """
        if not is_any_list(X[0]):
            X = [X]

        P_C = [[1 - self._theta.get(x[0], 0), self._theta.get(x[0], 0)] for x in X]
        return P_C
