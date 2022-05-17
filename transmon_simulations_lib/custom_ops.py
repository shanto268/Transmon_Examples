import qutip as qp
import numpy as np


def raising_op(dims):
    return qp.Qobj(np.diag(np.ones(dims-1), -1))


def lowering_op(dims):
    return qp.Qobj(np.diag(np.ones(dims-1), 1))
