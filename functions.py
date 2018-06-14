"""
A module that provides custom fnctions.
At time of writing, these functions are optimised and designed for the use-case
of the trpl.py script (time resolved photoluminescence analysis tool).
"""
import numpy as np

def testarray(*dims) :
    """
    Return a multidimensional array with increasing integer entries.

    Inputs:
    -------
    dims    :: tuple of int; the lengths along each dimension.
    """
    #ndim = len(dims)
    # Find the number of entries the resulting array will have
    n = 1
    for d in dims :
        n *= d
    
    # Create the base, 1-dimensional sequence
    res = np.arange(n)

    # Bring it to the desired dimension
    return res.reshape(dims)

def multiple_exponential(t, i=0, *A_TAU) :
    """
    Recursively build a function of the form

        f(t) = a1 * exp(-t/tau1) + a2 * exp(-t/tau2) + ... + an * exp(-t/taun)

    and return its value at t.
    A_TAU must have length 2n, where n is the number of exponentials
    you want in the sum. t can be any argument that could be passed to the 
    np.exp function.

    The reason some of this seems weirdly implemented is because this function
    has been made compatible with scipy.optimize.curve_fit which unpacks 
    parameters before passing them on to the fit function.
    Even now, some care has to be taken when passing arguments to this function
    directly - they probably have to be unpacked with the * operator, e.g.
        >>> multiple_exponential(np.array([1,2,3]), 0, *[0.5, 1, 0.5, 1])


    Inputs:
    -------
    t       :: array_like; input (x-values) to the resulting function.
    i       :: int; iteration level
    A_TAU   :: list of length 2n; the first n entries are amplitude 
                parameters for the different terms and the last n entries are 
                the decay times of each term
    """
    order = int(len(A_TAU)/2)
    i = int(i)
    if i < order-1 :
        return A_TAU[i] * np.exp(-t/A_TAU[order+i]) + multiple_exponential(t, i+1, *A_TAU)
    else :
        return A_TAU[i] * np.exp(-t/A_TAU[order+i])


def stretched_exponential(t, A, tau, h) :
    """
    Implement stretched exponentail decay:

        f(t) = A * exp( - (t/tau)^(1/h) )
    """
    return A * np.exp(-(t/tau)**(1./h))


def chi2(y, f, sigma, normalize=True) :
    """
    Return the chi-square value for a series of measured values y and 
    uncertainties sigma that are described by model f.

                    y_i - f_i
    chi2 = sum_i( (-----------)^2 )
                     sigma_i

    Can be normalized by the number of entries.

    Inputs:
    -------
    y       :: np.array; values of the data which is to be described
    f       :: np.array; values predicted by the model
    sigma   :: np.array or float; uncertainties on values y
    normalize
            :: boolean; whether to divide the result by the length of y or not

    """
    res = sum( ( (y-f)/sigma )**2 )
    res = res/len(y) if normalize else res
    return res
