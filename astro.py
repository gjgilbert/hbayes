import numpy as np
import pandas as pd
from   scipy import stats
from   scipy.interpolate import RectBivariateSpline

BIGG = 6.6743 * 10**(-8)   # Newton's constant; cm^3 / (g * s^2)

__all__ = ['calc_rho_star',
           'get_e_omega_obs_priors',
           'imp_sample_rhostar'
          ]


def calc_rho_star(P, T14, b, ror, ecc, omega):
    '''
    Inverting T14 equation from Winn 2010 
    
    Args:
        P: period in units of days
        T14: duration in units of days
        b: impact parameter
        ror: radius ratio
        ecc: eccentricity
        omega: argument of periastron in radians
    Out:
        rho_star: stellar density in units of g/cc
    '''
    per = P * 86400.
    dur = T14 * 86400.
    rho_star = 3*np.pi / (per**2 * BIGG) * (  (((1+ror)**2-b**2) / 
                                               np.sin((np.pi*dur/per) * (1+ecc*np.sin(omega))/(1-ecc**2))**2) + b**2)**(3/2)
    return rho_star


def get_e_omega_obs_priors(N, ecut):
    '''
    Get N random draws of ecc [0, ecut] and omega [-pi/2, 3pi/2],
    using the transit observability prior 
    (see: https://github.com/gjgilbert/notes/blob/main/calculate_e-omega_grid.ipynb)
    '''
    ngrid = 101
    ndraw = int(N)

    e_uni = np.linspace(0,ecut,ngrid)
    z_uni = np.linspace(0,1,ngrid)

    omega_grid = np.zeros((ngrid,ngrid))

    for i, e_ in enumerate(e_uni):
        x = np.linspace(-0.5*pi, 1.5*pi, int(1e4))
        y = (1 + e_*np.sin(x))/(2*pi)

        cdf = np.cumsum(y)
        cdf -= cdf.min()
        cdf = cdf/cdf.max()
        inv_cdf = interp1d(cdf, x)

        omega_grid[i] = inv_cdf(z_uni)

    RBS = RectBivariateSpline(e_uni, z_uni, omega_grid)

    e_draw = np.random.uniform(0, ecut, ndraw)
    z_draw = np.random.uniform(0, 1, ndraw)
    w_draw = RBS.ev(e_draw, z_draw)
    
    return e_draw, w_draw


def imp_sample_rhostar(samples, rho_star, norm=True, return_log=False, ecut=1.0, ew_obs_prior=False, distr='uniform', params=[]):
    '''
    Perform standard importance sampling from {IMPACT, ROR, PERIOD, DUR14} --> {ECC, OMEGA}
    
    Args
    ----
    samples [dataframe]: pandas dataframe of sampled data which includes: IMPACT, ROR, PERIOD, DUR14
    rho_star [tuple]: values of the true stellar density and its uncertainty
    norm [bool]: True to normalize weights before output (default=True)
    return_log [bool]: True to return ln(weights) instead of weights (default=False)
    ecut [float]: number between 0 and 1 indicating the upper bound on the ecc prior (default 1.0)
    ew_obs_prior [bool]: bool flag indicating whether or not to use the ecc-omega transit obs prior (default False)
    distr [str]: name of the distribution shape to sample ECC from; defaults to uniform
    params [list]: list of values to be used as parameters for the indicated distribution
    
    Output:
    weights [array]: importance sampling weights
    data [dataframe]: pandas dataframe containing all input and derived data, including: 
                      ECC: random values drawn from 0 to 'ecut' according to 'distr' and 'params'
                      OMEGA: random values drawn from -pi/2 to 3pi/2 (with transit obs prior if 'ew_obs_prior'=True)
                      IMPACT: inputs values
                      ROR: inputs values
                      PERIOD: inputs values
                      DUR14: inputs values
                      RHOSTAR: derived values
                      WEIGHTS (or LN_W): importance weights
    '''
    P   = samples.PERIOD.values
    T14 = samples.DUR14.values    
    ror = samples.ROR.values
    b   = samples.IMPACT.values
    
    N = len(b)

    if ew_obs_prior == True:
        ecc, omega = get_e_omega_obs_priors(N, ecut)

    else:
        if distr == 'uniform':
            ecc = np.random.uniform(0., ecut, N)

        elif distr == 'rayleigh':
            sigma = params[0]
            ecc = np.random.rayleigh(sigma, size=N)
            while np.any(ecc >= ecut):
                ecc[ecc >= ecut] = np.random.rayleigh(sigma, size=np.sum(ecc >= ecut))

        elif distr == 'beta':
            alpha_mu, beta_mu = params
            ecc = np.random.beta(alpha_mu, beta_mu, size=N)
            while np.any(ecc >= ecut):
                ecc[ecc >= ecut] = np.random.beta(alpha_mu, beta_mu, size=np.sum(ecc >= ecut))

        elif distr == 'half-gaussian':
            sigma = params[0]
            ecc = np.random.normal(loc=0, scale=sigma, size=N)
            while np.any((ecc >= ecut)|(ecc < 0)):
                ecc[(ecc >= ecut)|(ecc < 0)] = np.random.normal(loc=0, scale=sigma, size=np.sum((ecc >= ecut)|(ecc < 0)))

        omega = np.random.uniform(-0.5*np.pi, 1.5*np.pi, N)
        
        
    rho_samp = calc_rho_star(P, T14, b, ror, ecc, omega)
    log_weights = -0.5 * ((rho_samp - rho_star[0]) / rho_star[1]) ** 2
    
    # flag weights that are NaN-valued or below machine precision
    bad = np.isnan(log_weights) + (log_weights < np.log(np.finfo(float).eps))
    
    # prepare outputs
    data = pd.DataFrame()
    data['PERIOD']  = P[~bad]
    data['ROR']     = ror[~bad]
    data['IMPACT']  = b[~bad]
    data['DUR14']   = T14[~bad]
    data['ECC']     = ecc[~bad]
    data['OMEGA']   = omega[~bad]
    data['RHOSTAR'] = rho_samp[~bad]

    if return_log:       
        data['LN_WT'] = log_weights[~bad]
        return log_weights, data

    else:
        weights = np.exp(log_weights[~bad] - np.max(log_weights[~bad]))

        if norm:
            weights /= np.sum(weights)
        data['WEIGHTS'] = weights
        return weights, data

