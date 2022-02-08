import qutip as qp

class TmonEigensystem:
    def __init__(self, Ec=None, Ej=None, alpha=None, d=None, evecs=None,
                 evals=None, h_op=None, n_op=None):
        if (alpha is not None) and (d is not None):
            raise ValueError("only `alpha` or `d` can be supplied since "
                             "they are in one-to-one correspondence")
        self.Ec = Ec
        self.Ej = Ej
        if (alpha is not None) and (d is None):
            self.alpha = alpha
            self.d = (1-alpha)/(1+alpha)
        elif (d is not None) and (alpha is None):
            self.alpha = (1-d)/(1+d)
            self.d = d
        elif (d is None) and (alpha is None):
            raise ValueError("none of the asymmetry parameters "
                             "supplied: `d` or `alpha`")
        self.phi = 0
        self.evecs = evecs
        self.evals = evals - evals[0]
        self.Nc = h_op.shape[0]
        self.h_op = h_op
        self.n_op = n_op
        self.w01 = evals[1] - evals[0]
        self.w12 = evals[2] - evals[1] if len(evals) > 2 else None

    def _truncate(self, operator, n_trunc=None):
        if n_trunc is not None:
            pass
        else:
            n_trunc = self._N_trunc
        return qp.Qobj(operator[:n_trunc, :n_trunc])
