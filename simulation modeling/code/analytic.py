from math import exp


class analytic:

    def __init__(self, n, x0, Eh, El, a, alpha, h):
        self.h = h
        self.expo_const = alpha * (Eh - El) * ((1 / n) - a)
        self.c = (1/x0) - 1
        self.t = 0
        self.x0 = x0
        self.Eh = Eh
        self.El = El
        self.alpha = alpha
        self.a = a
        self.n = n


    def advance(self):
        self.t += self.h
        return exp(self.expo_const * self.t) / (exp(self.expo_const * self.t) + self.c)

    def slope(self, x):
        return self.alpha * (self.Eh - self.El) * (x * (1 - x)) * ((1/self.n) - self.a)
