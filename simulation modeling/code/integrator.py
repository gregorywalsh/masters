class integrator:

    @staticmethod
    def get_f(Eh, El, n, a, alpha):
        def f(x):
            return alpha * x * (1 - x) * (Eh - El) * ((1 / n) - a)
        return f

    def __init__(self, n, x0, Eh, El, a, alpha, h):
        self.f = integrator.get_f(Eh=Eh, El=El, n=n, a=a, alpha=alpha)
        self.x0 = x0
        self.x = x0
        self.h = h

    def advance(self, verbose=False):
        K1 = self.h * self.f(self.x)
        K2 = self.h * self.f(self.x + (0.5 * K1))
        K3 = self.h * self.f(self.x + (0.5 * K2))
        K4 = self.h * self.f(self.x + K3)
        self.x = self.x + (K1 + (2 * K2) + (2 * K3) + K4) / 6
        if verbose:
            return (self.x, K1, K2, K3, K4)
        else:
            return self.x
