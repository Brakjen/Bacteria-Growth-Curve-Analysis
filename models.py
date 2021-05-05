import numpy as np
import sympy as sp
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import quad


class BaseModel:
    """Base class for curve fitting models.

    The following models inherit from this class:

    - LogisticZero: Simple three-parameter logistic model
    - Logistic2k: Five-parameter logistic with two rate parameters
    - Logistic2kZero: Four-parameter version of above (lower asymptote at zero)
    - Richards: A Richards-type model (five parameters)
    - RichardsZero: Four-parameter Richards (lower asymptote at zero)
    - Gompertz: A Four-parameter Gompertz model
    - GompertzZero: A three-parameter Gompertz model (lower asymptote at zero)

    Some of these models work well for symmetric growth curves, but I find that
    the extra flexibility of the `2k` models (with two rate parameters) more often
    leads to qualitatively more correct fits. By this I mean that the inflection point
    on the fitted curve actually is located in the high-rate region of the data.
    The downside of the 2k models is that one loses the intuitive interpretation
    of the rate parameters, since the max rate must be computed numerically.
    (Or, quite possibly, there exists an analytical expression for the max growth
    rate in terms of the individual growth rates, but I have made no efforts into
    finding such an expression. As of now, I use scipy to find the roots of the
    second derivative of the fitted function, see details in the "max_growth_rate"
    methods.)

    The `Zero` models without a parameter for the lower asymptote, which in practice
    means that the lower asymptote always is zero. This might be logical for data that
    start at zero. One less parameter also speeds up the fitting.

    I have also tried to provide reasonable initial parameters for the individual
    models, and I have for some parameters restricted the optimization space to prevent
    unphysical fits (e.g. negative growth rates):

    Initial parameters
    - rates: average growth rate from t=tmin to t=tmax
             r0 = (max(y) - min(y)) / (max(x) - min(x))
    - upper asymptotes: The maximum y value
                        max(y)
    - lower asymptotes: The minimum y value
                        min(y)
    - Horizontal shift parameters: 1
    - Asymmetry parameter v (Richards models): 1
    """
    def __init__(self, x=None, y=None, maxfev=None, p0=None, lower=None, upper=None):
        self.x = x if x is not None else []
        self.y = y if y is not None else []

        self.maxfev = 2000 if maxfev is None else maxfev
        self.p0 = p0
        self.lower = lower
        self.upper = upper
        self.popt = None
        self.pcov = None
        self.vars = None

    def fit(self):
        """
        Performs curve fit on x and y data.

        Returns
        -------

        """
        popt, pcov = curve_fit(self.func, self.x, self.y, maxfev=self.maxfev,
                               p0=self.p0, bounds=(self.lower, self.upper))
        self.popt = tuple(popt)
        self.pcov = pcov
        self.vars = {v: self.popt[i] for i, v in enumerate(self.vars)}

    def area_under_curve(self, a=None, b=None):
        """

        Parameters
        ----------
        a : float
                Lower integration limit
        b : float
                Upper integration limit

        Returns
        -------
        float
            Area under the curve
        """
        if a is None:
            a = min(self.x)
        if b is None:
            b = max(self.x)

        I_1 = (b - a) * min(self.y)
        I_2, _ = quad(self.func, args=self.popt, a=a, b=b)
        return abs(I_1 - I_2)

    def func(self, *args):
        """To be overwritten in child class."""
        raise NotImplementedError

    def deriv1(self, *args):
        """To be overwritten in child class."""
        raise NotImplementedError

    def deriv2(self, *args):
        """To be overwritten in child class."""
        raise NotImplementedError

    def symbolic_func(self):
        """To be overwritten in child class."""
        raise NotImplementedError

    def symbolic_deriv1(self):
        """
        Symbolic representation of the first derivative.

        Returns
        -------
        sympy expression
        """
        return sp.diff(self.symbolic_func(), 'x')

    def symbolic_deriv2(self):
        """
        Symbolic representation of the first derivative.

        Returns
        -------
        sympy expression
        """
        return sp.diff(self.symbolic_deriv1(), 'x')

    def max_growth_rate(self):
        """To be overwritten in child class."""
        raise NotImplementedError

    def inflection_point(self):
        """To be overwritten in child class."""
        raise NotImplementedError


class LogisticZero(BaseModel):
    """
    Simple logistic model, with the lower asymptote at zero.

    f(x) = U/(1 + exp(-k*(x - x0)))

    Parameters
    -----------
    U : Upper asymptote
    k : Maximum growth rate
    x0 : x coordinate of maximum growth rate

    """
    def __init__(self, x=None, y=None, maxfev=None, p0=None, lower=None, upper=None):
        if not (x is None and y is None):
            if p0 is None:
                k0 = (max(y) - min(y)) / (max(x) - min(x))
                p0 = [max(y), k0, 1]

            if lower is None:
                lower = [0, 0, min(x)]

            if upper is None:
                upper = [np.inf, np.inf, max(x)]

        BaseModel.__init__(self, x=x, y=y, maxfev=maxfev, p0=p0, lower=lower, upper=upper)
        self.vars = ('U', 'k', 'x0')

    def func(self, x, U, k, x0):
        return U / (1 + np.exp(-k*(x-x0)))

    def max_growth_rate(self):
        return self.vars['k']

    def inflection_point(self):
        return self.vars['x0']

    def symbolic_func(self):
        x, U, k, x0 = sp.symbols('x U k x0')
        return U / (1 + sp.exp(-k*(x-x0)))

    def deriv1(self, x, U, k, x0):
        exp = np.exp(-k*(x-x0))
        return U * k * exp / (1 + exp)**2

    def deriv2(self, x, U, k, x0):
        exp = np.exp(-k*(x-x0))
        return 2*U*k*k * exp**2 / (1 + exp)**3 - U*k*k * exp / (1 + exp)**2


class Logistic2k(BaseModel):
    """
    Logistic model with two rate parameterrs.

    f(x) = L + (-L + U)/((1 + exp(-k1*(x - x0)))*(1 + exp(-k2*(x - x0))))

    Parameters
    -----------
    U : Upper asymptote
    L : Lower asymptote
    k1 : growth rate parameter
    k2 : growth rate parameter
    x0 : x coordinate of maximum growth rate

    """
    def __init__(self, x=None, y=None, maxfev=None, p0=None, lower=None, upper=None):
        if not (x is None and y is None):
            if p0 is None:
                k0 = (max(y) - min(y)) / (max(x) - min(x))
                p0 = [max(y), min(y), k0, k0, 1]

            if lower is None:
                lower = [-np.inf, -np.inf, 0, 0, min(x)]

            if upper is None:
                upper = [np.inf, np.inf, np.inf, np.inf, max(x)]

        BaseModel.__init__(self, x=x, y=y, maxfev=maxfev, p0=p0, lower=lower, upper=upper)
        self.vars = ('U', 'L', 'k1', 'k2', 'x0')

    def func(self, x, U, L, k1, k2, x0):
        return L + (U-L) / (1 + np.exp(-k1*(x-x0))) / (1 + np.exp(-k2*(x-x0)))

    def symbolic_func(self):
        x, U, L, k1, k2, x0 = sp.symbols('x U L k1 k2 x0')
        return L + (U - L) / (1 + sp.exp(-k1 * (x - x0))) / (1 + sp.exp(-k2 * (x - x0)))

    def deriv1(self, x, U, L, k1, k2, x0):
        exp1 = np.exp(-k1*(x-x0))
        exp2 = np.exp(-k2 * (x - x0))
        pre = U - L
        T1 = k1 * pre * exp1 / ( (1+exp1)**2 * (1+exp2) )
        T2 = k2 * pre * exp2 / ( (1+exp2)**2 * (1+exp1) )
        return T1 + T2

    def deriv2(self, x, U, L, k1, k2, x0):
        exp1 = np.exp(-k1 * (x - x0))
        exp2 = np.exp(-k2 * (x - x0))
        pre = U - L
        T1 = -k1**2 * pre * exp1 / ( (1+exp1)**2 * (1+exp2) )
        T2 = -k2**2 * pre * exp2 / ( (1+exp2)**2 * (1+exp1) )
        T3 = 2*k1**2 * pre * exp1**2 / ( (1+exp1)**3 * (1+exp2) )
        T4 = 2*k2**2 * pre * exp2**2 / ( (1+exp2)**3 * (1+exp1) )
        T5 = 2*k1*k2 * pre * exp1*exp2 / ( (1+exp1)**2 * (1+exp2)**2 )
        return T1 + T2 + T3 + T4 + T5

    def max_growth_rate(self):
        root = fsolve(self.deriv2, args=self.popt, x0=self.vars['x0'])
        return self.deriv1(root, *self.popt)[0]

    def inflection_point(self):
        return fsolve(self.deriv2, args=self.popt, x0=self.vars['x0'])[0]


class Logistic2kZero(BaseModel):
    """
    Logistic model with two rate parameterrs.

    f(x) = U/((1 + exp(-k1*(x - x0)))*(1 + exp(-k2*(x - x0))))

    Parameters
    -----------
    U
    k1
    k2
    x0

    """
    def __init__(self, x=None, y=None, maxfev=None, p0=None, lower=None, upper=None):
        if not (x is None and y is None):
            if p0 is None:
                k0 = (max(y) - min(y)) / (max(x) - min(x))
                p0 = [max(y), k0, k0, 1]

            if lower is None:
                lower = [-np.inf, 0, 0, min(x)]

            if upper is None:
                upper = [np.inf, np.inf, np.inf, max(x)]

        BaseModel.__init__(self, x=x, y=y, maxfev=maxfev, p0=p0, lower=lower, upper=upper)
        self.vars = ('U', 'k1', 'k2', 'x0')

    def func(self, x, U, k1, k2, x0):
        return U / (1 + np.exp(-k1*(x-x0))) / (1 + np.exp(-k2*(x-x0)))

    def symbolic_func(self):
        x, U, k1, k2, x0 = sp.symbols('x U k1 k2 x0')
        return U / (1 + sp.exp(-k1 * (x - x0))) / (1 + sp.exp(-k2 * (x - x0)))

    def deriv1(self, x, U, k1, k2, x0):
        exp1 = np.exp(-k1 * (x - x0))
        exp2 = np.exp(-k2 * (x - x0))
        T1 = k1 * U * exp1 / ((1 + exp1) ** 2 * (1 + exp2))
        T2 = k2 * U * exp2 / ((1 + exp2) ** 2 * (1 + exp1))
        return T1 + T2

    def deriv2(self, x, U, k1, k2, x0):
        exp1 = np.exp(-k1 * (x - x0))
        exp2 = np.exp(-k2 * (x - x0))
        T1 = -k1 ** 2 * U * exp1 / ((1 + exp1)**2 * (1 + exp2))
        T2 = -k2 ** 2 * U * exp2 / ((1 + exp2)**2 * (1 + exp1))
        T3 = 2 * k1 ** 2 * U * exp1 ** 2 / ((1 + exp1)**3 * (1 + exp2))
        T4 = 2 * k2 ** 2 * U * exp2 ** 2 / ((1 + exp2)**3 * (1 + exp1))
        T5 = 2 * k1 * k2 * U * exp1 * exp2 / ((1 + exp1)**2 * (1 + exp2)**2)
        return T1 + T2 + T3 + T4 + T5

    def max_growth_rate(self):
        root = fsolve(self.deriv2, args=self.popt, x0=self.vars['x0'])
        return self.deriv1(root, *self.popt)[0]

    def inflection_point(self):
        return fsolve(self.deriv2, args=self.popt, x0=self.vars['x0'])[0]


class Gompertz(BaseModel):
    """
    Gompertz model

    f(x) = L + (-L + U)*exp(-exp(-k*(x - x0)))

    Parameters
    -----------
    U
    L
    k
    x0

    """
    def __init__(self, x=None, y=None, maxfev=None, p0=None, lower=None, upper=None):
        if not (x is None and y is None):
            if p0 is None:
                k0 = (max(y) - min(y)) / (max(x) - min(x))
                p0 = [max(y), min(y), k0, 1]

            if lower is None:
                lower = [-np.inf, -np.inf, 0, min(x)]

            if upper is None:
                upper = [np.inf, np.inf, np.inf, max(x)]

        BaseModel.__init__(self, x=x, y=y, maxfev=maxfev, p0=p0, lower=lower, upper=upper)
        self.vars = ('U', 'L', 'k', 'x0')

    def func(self, x, U, L, k, x0):
        return L + (U - L) * np.exp(-np.exp(-k*(x-x0)))

    def symbolic_func(self):
        x, U, L, k, x0 = sp.symbols('x U L k x0')
        return L + (U - L) * sp.exp(-sp.exp(-k * (x - x0)))

    def max_growth_rate(self):
        return self.vars['k']

    def inflection_point(self):
        return self.vars['x0']

    def deriv1(self, x, U, L, k, x0):
        exp = np.exp(-k*(x-x0))
        return k*(U-L) * exp * np.exp(-exp)

    def deriv2(self, x, U, L, k, x0):
        exp = np.exp(-k * (x - x0))
        return k*k*(U-L)*exp * (exp * np.exp(-exp) - np.exp(-exp))


class GompertzZero(BaseModel):
    """
    Gompertz model

    f(x) = U*exp(-exp(-k*(x - x0)))

    Parameters
    -----------
    U
    k
    x0

    """
    def __init__(self, x=None, y=None, maxfev=None, p0=None, lower=None, upper=None):
        if not (x is None and y is None):
            if p0 is None:
                k0 = (max(y) - min(y)) / (max(x) - min(x))
                p0 = [max(y), k0, 1]

            if lower is None:
                lower = [-np.inf, 0, min(x)]

            if upper is None:
                upper = [np.inf, np.inf, max(x)]

        BaseModel.__init__(self, x=x, y=y, maxfev=maxfev, p0=p0, lower=lower, upper=upper)
        self.vars = ('U', 'k', 'x0')

    def func(self, x, U, k, x0):
        return U * np.exp(-np.exp(-k*(x-x0)))

    def symbolic_func(self):
        x, U, k, x0 = sp.symbols('x U k x0')
        return U * sp.exp(-sp.exp(-k * (x - x0)))

    def max_growth_rate(self):
        return self.vars['k']

    def inflection_point(self):
        return self.vars['x0']

    def deriv1(self, x, U, k, x0):
        exp = np.exp(-k*(x-x0))
        return k*U * exp * np.exp(-exp)

    def deriv2(self, x, U, k, x0):
        exp = np.exp(-k * (x - x0))
        return k*k*U*exp * (exp * np.exp(-exp) - np.exp(-exp))


class Richards(BaseModel):
    """
    Richards model

    f(x) = L + (-L + U)*(1 + exp(-k*(x - x0)))**(-1/v)

    Parameters
    -----------
    U
    L
    k
    v
    x0

    """
    def __init__(self, x=None, y=None, maxfev=None, p0=None, lower=None, upper=None):
        if not (x is None and y is None):
            if p0 is None:
                k0 = (max(y) - min(y)) / (max(x) - min(x))
                p0 = [max(y), min(y), k0, 0.5, 1]

            if lower is None:
                lower = [-np.inf, -np.inf, 0, 0, min(x)]

            if upper is None:
                upper = [np.inf, np.inf, np.inf, np.inf, max(x)]

        BaseModel.__init__(self, x=x, y=y, maxfev=maxfev, p0=p0, lower=lower, upper=upper)
        self.vars = ('U', 'L', 'k', 'v', 'x0')

    def func(self, x, U, L, k, v, x0):
        return L + (U - L) / ((1 + np.exp(-k*(x-x0)))**(1/v))

    def symbolic_func(self):
        x, U, L, k, v, x0 = sp.symbols('x U L k v x0')
        return L + (U - L) / ((1 + sp.exp(-k*(x-x0)))**(1/v))

    def max_growth_rate(self):
        return self.vars['k']

    def inflection_point(self):
        return fsolve(self.deriv2, args=self.popt, x0=self.vars['x0'], maxfev=2000)[0]

    def deriv1(self, x, U, L, k, v, x0):
        exp = np.exp(-k*(x-x0))
        return k*(1+exp)**(-1/v) * (U-L)*exp / (v*(1+exp))

    def deriv2(self, x, U, L, k, v, x0):
        exp = np.exp(-k * (x - x0))
        pre = U-L
        return k**2 * (1+exp)**(-1/v) * pre * exp / (v * (1+exp)) * (-1 + exp/(v*(1+exp)) + exp/(v**2 * (1+exp)))


class RichardsZero(BaseModel):
    """
    Richards model

    f(x) = U*(1 + exp(-k*(x - x0)))**(-1/v)

    Parameters
    -----------
    U
    k
    v
    x0

    """
    def __init__(self, x=None, y=None, maxfev=None, p0=None, lower=None, upper=None):
        if not (x is None and y is None):
            if p0 is None:
                k0 = (max(y) - min(y)) / (max(x) - min(x))
                p0 = [max(y), k0, 1, 1]

            if lower is None:
                lower = [-np.inf, 0, 0, min(x)]

            if upper is None:
                upper = [np.inf, np.inf, np.inf, max(x)]

        BaseModel.__init__(self, x=x, y=y, maxfev=maxfev, p0=p0, lower=lower, upper=upper)
        self.vars = ('U', 'k', 'v', 'x0')

    def func(self, x, U, k, v, x0):
        return U / ((1 + np.exp(-k*(x-x0)))**(1/v))

    def symbolic_func(self):
        x, U, k, v, x0 = sp.symbols('x U k v x0')
        return U / ((1 + sp.exp(-k * (x - x0))) ** (1 / v))

    def max_growth_rate(self):
        return self.vars['k']

    def inflection_point(self):
        return fsolve(self.deriv2, args=self.popt, x0=self.vars['x0'], maxfev=2000)[0]

    def deriv1(self, x, U, k, v, x0):
        exp = np.exp(-k * (x - x0))
        return k * (1 + exp) ** (-1 / v) * U * exp / (v * (1 + exp))

    def deriv2(self, x, U, k, v, x0):
        exp = np.exp(-k * (x - x0))
        T1 = -k ** 2 * (1 + exp) ** (-1 / v) * U * exp / (v * (1 + exp))
        T2 = k ** 2 * (1 + exp) ** (-1 / v) * U * exp ** 2 / (v * (1 + exp) ** 2)
        T3 = k ** 2 * (1 + exp) ** (-1 / v) * U * exp ** 2 / (v ** 2 * (1 + exp) ** 2)
        return T1 + T2 + T3
