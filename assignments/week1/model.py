import numpy as np


class LinearRegression:
    """
    Linear regression class implementing the analytical solution
    """

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model according to the given training data.

        Args:
            X (array of shape (n_samples, n_features)): Predictors for the linear model
            y (array of shape (n_samples, 1)): Target for training the linear model
        """

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        X_new = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        w_b = np.linalg.inv(X_new.T @ X_new) @ (X_new.T @ y)
        self.w, self.b = w_b[:-1, :], w_b[-1:, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model.

        Args:
            X (array of shape (n_samples, n_features)): Predictors for the linear model

        Returns:
            y_pred (array of shape (n_samples, 1)): Returns predicted values.
        """

        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    # init of base is called automatically if not implemented in child
    # if implemented in child and we want to run parent's init
    # then we have to explicitly call it in child using super().__init__()

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """Fit the model according to the given training data using gradient descent algorithm.

        Args:
            X (array of shape (n_samples, n_features)): Predictors for the linear model
            y (array of shape (n_samples, 1)): Target for training the linear model
            lr (float, optional): Learning rate of gradient descent. Defaults to 0.01.
            epochs (int, optional): Number of epochs for gradient descent. Defaults to 1000.
        """

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        X_new = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        w_b = np.zeros((X_new.shape[1], 1))
        for i in range(epochs):
            y_hat = X_new @ w_b
            grad = (-2 * X_new.T @ (y - y_hat)) / X_new.shape[0]
            w_b = w_b - lr * grad

        self.w, self.b = w_b[:-1, :], w_b[-1:, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Args:
            X (array of shape (n_samples, n_features)): Predictors for the linear model

        Returns:
            y_pred (array of shape (n_samples, 1)): Returns predicted values.
        """

        return X @ self.w + self.b
