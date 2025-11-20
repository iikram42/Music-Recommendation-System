import numpy as np
from src.model.matrix_factorization import MatrixFactorization
import matplotlib.pyplot as plt

def run_lr_experiment(interactions, n_users, n_items, lrs=[0.001,0.01,0.05]):
    histories = {}
    for lr in lrs:
        mf = MatrixFactorization(n_users, n_items, lr=lr, reg=0.01)
        loss = mf.train(interactions, epochs=5, batch_size=32, verbose=False)
        histories[lr] = loss
    return histories

def plot_histories(histories, out="experiments/plots/lr_plot.png"):
    for lr,loss in histories.items():
        plt.plot(loss, label=f"lr={lr}")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Rate Comparison")
    plt.savefig(out)
    plt.close()
