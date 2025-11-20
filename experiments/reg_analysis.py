import numpy as np
from src.model.matrix_factorization import MatrixFactorization
import matplotlib.pyplot as plt

def run_reg_experiment(interactions, n_users, n_items, regs=[0.0,0.01,0.1]):
    histories = {}
    for reg in regs:
        mf = MatrixFactorization(n_users, n_items, lr=0.01, reg=reg)
        loss = mf.train(interactions, epochs=5, batch_size=32, verbose=False)
        histories[reg] = loss
    return histories

def plot_histories(histories, out="experiments/plots/reg_plot.png"):
    for reg,loss in histories.items():
        plt.plot(loss, label=f"reg={reg}")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Regularization Comparison")
    plt.savefig(out)
    plt.close()
