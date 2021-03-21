import numpy as np


class FactorEstimator:
    def __init__(self, residual):
        self._residual = residual
        self.T = residual.shape[0]
        self.N = residual.shape[1]

    def r_hat(self, rmax, panelty, id):
        assert id in list(range(1, 5)), "id shoud be in 1~4"
        r_range = range(1, rmax + 1)
        if panelty == "IC":
            v = [self._calculate_ic(r, id) for r in r_range]
            return r_range[np.argmin(v)]
        elif panelty == "PC":
            v = [self._calculate_pc(r, rmax, id) for r in r_range]
            return r_range[np.argmin(v)]
        else:
            raise ValueError("panelty should be 'IC' or 'PC'")

    def _calculate_ic(self, r, id):
        f_hat = self._calculate_f_tilde(r)
        lambda_hat = self._calculate_lambda_tilde(f_hat, r)
        vkf = self._calculate_vkf(lambda_hat, f_hat)
        ic_p = np.log(vkf) + r * self._calculate_g(id)
        return ic_p

    def _calculate_pc(self, r, rmax, id):
        f_hat = self._calculate_f_tilde(r)
        lambda_hat = self._calculate_lambda_tilde(f_hat, r)
        f_hat_rmax = self._calculate_f_tilde(rmax)
        lambda_hat_rmax = self._calculate_lambda_tilde(f_hat_rmax, rmax)
        vkf = self._calculate_vkf(lambda_hat, f_hat)
        sigma_sq = self._calculate_vkf(lambda_hat_rmax, f_hat_rmax)
        pc = vkf + r * sigma_sq * self._calculate_g(id)
        return pc

    def _calculate_f_tilde(self, r):
        uut = np.zeros(shape=(self.T, self.T))
        for i in range(self.N):
            uut = uut + self._residual[:, i, np.newaxis].dot(
                self._residual[:, i, np.newaxis].T
            )
        w, v = np.linalg.eig(uut)
        f_hat = np.sqrt(self.T) * v[:, np.argsort(-w)[0:r]]
        return f_hat

    def _calculate_lambda_tilde(self, f_hat, r):
        return np.array(
            [f_hat.T.dot(self._residual[:, i]) / self.T for i in range(self.N)]
        )

    def _calculate_vkf(self, lambda_hat, f_hat):
        vkf = ((self._residual.T - lambda_hat.dot(f_hat.T)) ** 2).sum() / (
            self.N * self.T
        )
        return vkf

    def _calculate_g(self, id):
        g = {
            1: lambda: (self.N + self.T)
            / (self.N * self.T)
            * np.log((self.N * self.T) / (self.N + self.T)),
            2: lambda: (self.N + self.T)
            / (self.N * self.T)
            * np.log(min(self.N, self.T)),
            3: lambda: (np.log(min(self.N, self.T))) / min(self.N, self.T),
            4: lambda: 2 / self.T,
        }[id]()
        return g
