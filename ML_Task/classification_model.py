# classification_model.py

from sklearn.tree import DecisionTreeClassifier


def train_model(X_train, y_train):
    """
    Train a Decision Tree Classification model.
    """

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    return model