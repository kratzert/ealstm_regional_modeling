"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import numpy as np


def calc_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe-Effiency
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    
    Returns
    -------
    float
        Nash-Sutcliffe-Efficiency
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If all values in the observations are equal
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    # denominator of the fraction term
    denominator = np.sum((obs - np.mean(obs))**2)

    # this would lead to a division by zero error and nse is defined as -inf
    if denominator == 0:
        msg = [
            "The Nash-Sutcliffe-Efficiency coefficient is not defined ",
            "for the case, that all values in the observations are equal.",
            " Maybe you should use the Mean-Squared-Error instead."
        ]
        raise RuntimeError("".join(msg))

    # numerator of the fraction term
    numerator = np.sum((sim - obs)**2)

    # calculate the NSE
    nse_val = 1 - numerator / denominator

    return nse_val


def calc_alpha_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Alpha decomposition of the NSE, see Gupta et al. 2009

    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations

    Returns
    -------
    float
        Alpha decomposition of the NSE

    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    return np.std(sim) / np.std(obs)


def calc_beta_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Beta decomposition of NSE. See Gupta et. al 2009
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations

    Returns
    -------
    float
        Beta decomposition of the NSE

    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    return (np.mean(sim) - np.mean(obs)) / np.std(obs)


def calc_fdc_fms(obs: np.ndarray, sim: np.ndarray, m1: float = 0.2, m2: float = 0.7) -> float:
    """[summary]
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    m1 : float, optional
        Lower bound of the middle section. Has to be in range(0,1), by default 0.2
    m2 : float, optional
        Upper bound of the middle section. Has to be in range(0,1), by default 0.2
    
    Returns
    -------
    float
        Bias of the middle slope of the flow duration curve (Yilmaz 2018).
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If `m1` is not in range(0,1)
    RuntimeError
        If `m2` is not in range(0,1)
    RuntimeError
        If `m1` >= `m2`
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    if (m1 <= 0) or (m1 >= 1):
        raise RuntimeError("m1 has to be in the range (0,1)")

    if (m2 <= 0) or (m2 >= 1):
        raise RuntimeError("m1 has to be in the range (0,1)")

    if m1 >= m2:
        raise RuntimeError("m1 has to be smaller than m2")

    # for numerical reasons change 0s to 1e-6
    sim[sim == 0] = 1e-6
    obs[obs == 0] = 1e-6

    # sort both in descending order
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # calculate fms part by part
    qsm1 = np.log(sim[np.round(m1 * len(sim)).astype(int)] + 1e-6)
    qsm2 = np.log(sim[np.round(m2 * len(sim)).astype(int)] + 1e-6)
    qom1 = np.log(obs[np.round(m1 * len(obs)).astype(int)] + 1e-6)
    qom2 = np.log(obs[np.round(m2 * len(obs)).astype(int)] + 1e-6)

    fms = ((qsm1 - qsm2) - (qom1 - qom2)) / (qom1 - qom2 + 1e-6)

    return fms * 100


def calc_fdc_fhv(obs: np.ndarray, sim: np.ndarray, h: float = 0.02) -> float:
    """Peak flow bias of the flow duration curve (Yilmaz 2018).
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    h : float, optional
        Fraction of the flows considered as peak flows. Has to be in range(0,1), by default 0.02
    
    Returns
    -------
    float
        Bias of the peak flows
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If `h` is not in range(0,1)
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    if (h <= 0) or (h >= 1):
        raise RuntimeError("h has to be in the range (0,1)")

    # sort both in descending order
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # subset data to only top h flow values
    obs = obs[:np.round(h * len(obs)).astype(int)]
    sim = sim[:np.round(h * len(sim)).astype(int)]

    fhv = np.sum(sim - obs) / (np.sum(obs) + 1e-6)

    return fhv * 100


def calc_fdc_flv(obs: np.ndarray, sim: np.ndarray, l: float = 0.7) -> float:
    """[summary]
    
    Parameters
    ----------
    obs : np.ndarray
        Array containing the discharge observations
    sim : np.ndarray
        Array containing the discharge simulations
    l : float, optional
        Upper limit of the flow duration curve. E.g. 0.7 means the bottom 30% of the flows are 
        considered as low flows, by default 0.7
    
    Returns
    -------
    float
        Bias of the low flows.
    
    Raises
    ------
    RuntimeError
        If `obs` and `sim` don't have the same length
    RuntimeError
        If `l` is not in the range(0,1)
    """
    # make sure that metric is calculated over the same dimension
    obs = obs.flatten()
    sim = sim.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    if (l <= 0) or (l >= 1):
        raise RuntimeError("l has to be in the range (0,1)")

    # for numerical reasons change 0s to 1e-6
    sim[sim == 0] = 1e-6
    obs[obs == 0] = 1e-6

    # sort both in descending order
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # subset data to only top h flow values
    obs = obs[np.round(l * len(obs)).astype(int):]
    sim = sim[np.round(l * len(sim)).astype(int):]

    # transform values to log scale
    obs = np.log(obs + 1e-6)
    sim = np.log(sim + 1e-6)

    # calculate flv part by part
    qsl = np.sum(sim - sim.min())
    qol = np.sum(obs - obs.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return flv * 100
