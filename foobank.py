import pandas as pd
import numpy as np
import math

# GENERIC MATH

def calc_volume(diameter: float, length: float) -> float:
    '''
    calculates the volume of a cylinder
    '''
    return np.pi * (diameter / 2)**2 * length

def calc_dV(diameter: float, length: float, N: int) -> float:
    '''
    Calculates the differential volume.
    N is the number of integration steps.
    '''
    return calc_volume(diameter, length) / N

def interaction(km1i, km1im1, factor):
    value = km1i - factor * (km1i - km1im1)
    return value

def calc_factor(Q: float, dV: float, dt: float) -> float:
    '''
    Calculates propagation factor
    Volumetric flowrate / volume of section * delta time
    '''
    factor = Q / dV * dt
    if (factor > 1.0): print(f"Warning: factor exceeds 1: {factor}")
    return factor

# GAS SIMULATIONS

def propagate(
    t: np.ndarray,
    inlet: np.ndarray,
    dV: np.ndarray,
    Q: np.ndarray,
    IC: float = 0,
) -> np.ndarray:
    '''
    Propagates an isolated gas at a volumetric flowrate (Q) through arbitraty volume differentials (dV).
    t: nx1 array cumulative time vector [0, 1, 2.3, 4.5, etc.]
    inlet: nx1 array representing the function of the inflow of gas
    dV: mx1 array of the volume size of each section m
    Q: mx1 array of the volumetric flowrate of each section m
    IC: float representing the initial concentration of the gas in each section of the reactor at the start
    '''

    nTimesteps = t.shape[0]
    nSections = dV.shape[0]
    
    # nxm representing the concentration of a gas in all sections (m) over time (n)
    C = np.ones(shape=[nTimesteps, nSections]) * IC

    for k in range(nTimesteps): # iterate through time
        if (k==0): continue # skip first timestep
        
        dt = t[k] - t[k-1]

        for i in range(nSections): # iterate through sections
            thisSection = C[k-1,i]
            previousSection = C[k-1,i-1] if(i!=0) else inlet[k]
            
            factor = calc_factor(Q=Q[i], dV=dV[i], dt=dt)
            if (factor > 1):
                print(factor)
            C[k,i] = thisSection - factor * (thisSection - previousSection)
            # print(f"{C[k,i]} at time {t[k]}, section {i}")
            # print("")
        
    return C
