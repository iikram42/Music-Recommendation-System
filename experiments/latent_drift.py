import numpy as np
import matplotlib.pyplot as plt
from src.model.matrix_factorization import MatrixFactorization

def compute_drift(interactions, n_users, n_items):
    mf = MatrixFactorization(n_users, n_items)
    initial = mf.U.copy()
    mf.train(interactions, epochs=3, batch_size=32, verbose=False)
    final = mf.U
    drift = np.linalg.norm(final-initial, axis=1)
    return drift

def plot_drift(drift, out="experiments/plots/drift.png"):
    plt.bar(range(len(drift)), drift)
    plt.xlabel("User")
    plt.ylabel("Latent Vector Drift")
    plt.title("Latent Drift")
    plt.savefig(out)
    plt.close()
