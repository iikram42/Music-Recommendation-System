import numpy as np
import matplotlib.pyplot as plt
from src.model.matrix_factorization import MatrixFactorization

# simplistic gradient norm tracking extension
def run_gradient_norms(interactions, n_users, n_items):
    mf = MatrixFactorization(n_users, n_items, lr=0.01, reg=0.01)
    norms=[]
    for epoch in range(5):
        np.random.shuffle(interactions)
        for (u,i,r) in interactions[:200]:
            pred = mf.predict(u,i)
            e=r-pred
            grad_u = -2 * e * mf.V[i,:] + 2 * mf.reg * mf.U[u,:]
            norms.append(np.linalg.norm(grad_u))
            mf.sgd_update(u,i,r)
    return norms

def plot_norms(norms, out="experiments/plots/grad_norms.png"):
    plt.plot(norms)
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Tracking")
    plt.savefig(out)
    plt.close()
