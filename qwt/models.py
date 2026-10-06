"""The walk classifier of thesis Section 4.4 and its quantum comparator (Liu et al.'s kernel, eq. lat-kernel).

Walk QNN. Input: the exponent a = log_g x, i.e. the output of the discrete-logarithm step.
  1. the top h bits of a select the vertex v = floor(a 2^h / (p-1)) of the cycle C_{2^h};
  2. the position register evolves under the walk for time t (CTWalkQNN: e^{-itA}; HadamardWalkQNN: t coined steps);
  3. the position is measured; f(a) = sum_u w_u P_t(u | v) + b, the expectation of a trainable diagonal observable,
     and the predicted label is sgn f.
The mean-squared-error loss is quadratic in (w, b) and is minimised exactly, with a ridge penalty on w and the bias
unpenalised. t = 0 is the no-walk ablation (a read-out of the start vertex only).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from .walks import ctqw_distribution, hadamard_distribution


def _bins(X, p, h):
    a = np.asarray(X).reshape(-1).astype(np.int64)
    return (a * 2 ** h) // (p - 1)


class HadamardWalkQNN(BaseEstimator, ClassifierMixin):
    """Walk QNN with the coined Hadamard walk (t integer steps). X: column vector of exponents a."""

    def __init__(self, p=251, h=4, t=4, ridge=1e-2, shots=None, random_state=0):
        self.p, self.h, self.t, self.ridge, self.shots, self.random_state = p, h, t, ridge, shots, random_state

    def _distribution(self):
        return hadamard_distribution(self.h, self.t)

    def _probs(self, X):
        """(B, N) array of P_t(u | v) for the bins v of the exponents in X."""
        N = 2 ** self.h
        v = _bins(X, self.p, self.h)
        return self._distribution()[(np.arange(N)[None, :] - v[:, None]) % N]

    def fit(self, X, y):
        """Primal normal equations; ridge on w, bias unpenalised."""
        y = np.asarray(y, float)
        self.classes_ = np.array([-1, 1])
        Phi = np.c_[self._probs(X), np.ones(len(y))]
        R = self.ridge * np.eye(Phi.shape[1]); R[-1, -1] = 0
        self.coef_ = np.linalg.solve(Phi.T @ Phi + R, Phi.T @ y)
        return self

    def decision_function(self, X):
        P = self._probs(X)
        if self.shots:                                           # finite-shot estimate of <O_w>
            rng = np.random.default_rng(self.random_state)
            P = np.array([rng.multinomial(self.shots, q / q.sum()) for q in P]) / self.shots
        return P @ self.coef_[:-1] + self.coef_[-1]

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class CTWalkQNN(HadamardWalkQNN):
    """Walk QNN with the continuous-time walk e^{-itA} on C_{2^h}; t is continuous. The model of the thesis."""

    def _distribution(self):
        return ctqw_distribution(self.h, self.t)

    def fit(self, X, y):
        # the same least-squares problem; scikit-learn's Ridge solves it in the dual (kernel) form when 2^h
        # exceeds the number of samples and in the primal form otherwise
        from sklearn.linear_model import Ridge
        self.classes_ = np.array([-1, 1])
        r = Ridge(alpha=self.ridge).fit(self._probs(X), np.asarray(y, float))
        self.coef_ = np.r_[r.coef_, r.intercept_]
        return self


def dual_fit_predict(K_tr, K_te, y, lam):
    """Ridge with an unpenalised intercept in dual form, from the Gram matrices of the walk features
    (the walk kernel K_t(v, v') = sum_u P_t(u|v) P_t(u|v'), eq. walk-kernel); returns decision values."""
    m_tr = K_tr.mean(1); mm = K_tr.mean()
    Kc = K_tr - m_tr[:, None] - m_tr[None, :] + mm
    alpha = np.linalg.solve(Kc + lam * np.eye(len(y)), y - y.mean())
    Kt = K_te - K_te.mean(1)[:, None] - m_tr[None, :] + mm
    return Kt @ alpha + y.mean()


class LATKernelSVM(BaseEstimator, ClassifierMixin):
    """Liu-Arunachalam-Temme kernel: |phi_k(x)> = 2^{-k/2} sum_{i<2^k} |x g^i>,
    K = |<phi(x)|phi(x')>|^2 = (max(0, 2^k - d) / 2^k)^2, d the circular distance between the exponents."""

    def __init__(self, p=251, k=6, C=1.0):
        self.p, self.k, self.C = p, k, C

    def _K(self, A, B):
        A = np.asarray(A).reshape(-1, 1); B = np.asarray(B).reshape(1, -1)
        d = np.abs(A - B) % (self.p - 1)
        d = np.minimum(d, self.p - 1 - d)
        W = 2 ** self.k
        return (np.maximum(0, W - d) / W) ** 2

    def fit(self, X, y):
        self.X_ = np.asarray(X).reshape(-1)
        self.classes_ = np.array([-1, 1])
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(self._K(self.X_, self.X_), y)
        return self

    def predict(self, X):
        return self.svc_.predict(self._K(X, self.X_))
