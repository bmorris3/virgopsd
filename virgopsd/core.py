import numpy as np
import celerite
from celerite import terms
import astropy.units as u

__all__ = ['generate_fluxes', 'kernel']

parameter_vector = np.loadtxt('parameter_vector.txt')

nterms = len(parameter_vector)//3

kernel = terms.SHOTerm(log_S0=0, log_omega0=0, log_Q=0) 

for term in range(nterms-1): 
    kernel += terms.SHOTerm(log_S0=0, log_omega0=0, log_Q=0)

kernel.set_parameter_vector(parameter_vector)

gp = celerite.GP(kernel)

@u.quantity_input(cadence=u.s)
def generate_fluxes(size, cadence=60*u.s):
    """
    Generate an array of fluxes with zero mean which mimic the power spectrum of
    the SOHO/VIRGO SPM observations.
    
    Parameters
    ----------
    size : int
        Number of fluxes to generate
    cadence : `~astropy.units.Quantity`
        Length of time between fluxes
    
    Returns
    -------
    y : `~numpy.ndarray`
        Array of fluxes at cadence ``cadence`` of length ``size``.
    """
    x = np.arange(0, size//500, cadence.to(u.s).value) 
    gp.compute(x, check_sorted=False)

    y = gp.sample(500)
    
    y_concatenated = []

    for i, yi in enumerate(y): 
        xi = np.arange(len(yi))
        fit = np.polyval(np.polyfit(xi - xi.mean(), yi, 1), xi-xi.mean())
        yi -= fit

        if i == 0: 
            y_concatenated.append(yi)
        else: 
            offset = yi[0] - y_concatenated[i-1][-1]
            y_concatenated.append(yi - offset)
    y_concatenated = np.hstack(y_concatenated)
    
    x_c = np.arange(len(y_concatenated))
    
    y_concatenated -= np.polyval(np.polyfit(x_c - x_c.mean(), y_concatenated, 1), x_c - x_c.mean())
    
    return y_concatenated
