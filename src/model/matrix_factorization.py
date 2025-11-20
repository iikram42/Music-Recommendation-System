import numpy as np
import math

class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=20, lr=0.01, reg=0.01):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.U = 0.1 * np.random.randn(n_users, n_factors)
        self.V = 0.1 * np.random.randn(n_items, n_factors)

    def predict(self, u, i):
        return self.U[u, :].dot(self.V[i, :].T)

    def sgd_update(self, u, i, r):
        pred = self.predict(u, i)
        e = r - pred
        # gradients
        grad_u = -2 * e * self.V[i, :] + 2 * self.reg * self.U[u, :]
        grad_v = -2 * e * self.U[u, :] + 2 * self.reg * self.V[i, :]
        # update
        self.U[u, :] -= self.lr * grad_u
        self.V[i, :] -= self.lr * grad_v
        return e

    def train(self, interactions, epochs=10, batch_size=100, verbose=True):
        # interactions: list of (u, i, r)
        n = len(interactions)
        loss_history = []
        for epoch in range(epochs):
            np.random.shuffle(interactions)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch = interactions[start:start+batch_size]
                for (u,i,r) in batch:
                    e = self.sgd_update(u, i, r)
                    epoch_loss += e*e
            loss = epoch_loss / n
            loss_history.append(loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.6f}")
        return loss_history
