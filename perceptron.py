import numpy as np

class Perceptron:
    def __init__(self, input_dim, init_type="small_random"):
        if init_type == "zeros":
            self.w = np.zeros(input_dim)
        elif init_type == "large_random":
            self.w = np.random.randn(input_dim) * 10
        else:
            self.w = np.random.randn(input_dim) * 0.01

        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z = X @ self.w + self.b
        y_pred = self.sigmoid(z)
        return y_pred

    def compute_loss(self, y_true, y_pred):
        eps = 1e-15
        L = - (y_true * np.log(y_pred+eps) + (1 - y_true)*(np.log(1-y_pred+eps)))
        return np.mean(L)

    def fit(self, X_train, y_train, X_val, y_val, epochs, lr=0, batch_size=32):
        train_loss = []
        test_loss = []

        for i in range(epochs):
            ind = np.random.permutation(X_train.shape[0])

            X_train = X_train[ind]
            y_train = y_train[ind]

            for st  in range(0, X_train.shape[0], batch_size):
                end = st + batch_size

                X_batch = X_train[st:end]
                y_batch = y_train[st:end]

                y_pred = self.forward(X_batch)

                self.w = self.w - lr * (X_batch.T @ (y_pred - y_batch) / X_batch.shape[0])
                self.b = self.b - lr * np.mean(y_pred - y_batch)

            y_pred_train = self.forward(X_train)
            y_pred_val = self.forward(X_val)

            train_loss.append( self.compute_loss(y_train, y_pred_train))
            test_loss.append(self.compute_loss(y_val, y_pred_val))

        return train_loss, test_loss

    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred >= 0.5).astype(int)