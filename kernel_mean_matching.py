import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy.linalg import norm
from dataset_parsing import get_data_response, MERGED_DATASET
from model_research import parse_data_per_iter
from cvxopt import solvers, matrix
import sklearn.metrics.pairwise as sk
import math


def kernel(x_s: np.array, x_t: np.array, n_s: int, n_t: int, sigma: float) -> float:
    delta = norm(x_s - x_t)
    gaussian = np.exp(- delta / (2 * sigma))
    return n_s / n_t * gaussian


def kmm(x_train, x_test, sigma):
    n_tr = len(x_train)
    n_te = len(x_test)
    solvers.options["show_progress"] = False
    # calculate Kernel
    K_ns = sk.rbf_kernel(x_train, x_train, sigma)
    # make it symmetric
    K = 0.9 * (K_ns + K_ns.transpose())

    # calculate kappa
    kappa_r = sk.rbf_kernel(x_train, x_test, sigma)
    ones = np.ones(shape=(n_te, 1))
    kappa = np.dot(kappa_r, ones)
    kappa = -(float(n_tr) / float(n_te)) * kappa

    # calculate eps
    eps = (math.sqrt(n_tr) - 1) / math.sqrt(n_tr)

    # constraints
    A0 = np.ones(shape=(1, n_tr))
    A1 = -np.ones(shape=(1, n_tr))
    A = np.vstack([A0, A1, - np.eye(n_tr), np.eye(n_tr)])
    b = np.array([[n_tr * (eps + 1), n_tr * (eps - 1)]])
    b = np.vstack([b.T, - np.zeros(shape=(n_tr, 1)), np.ones(shape=(n_tr, 1)) * 1000])

    P = matrix(K, tc='d')
    q = matrix(kappa, tc='d')
    G = matrix(A, tc='d')
    h = matrix(b, tc='d')
    beta = solvers.qp(P, q, G, h)
    return [i for i in beta['x']]
