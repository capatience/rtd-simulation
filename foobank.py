import pandas as pd
import numpy as np

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
    verbose: bool = False
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

        if (verbose):
            progress = k/nTimesteps*100
            if (progress%5 == 0):
                print(f"Simulating... {int(progress)}")
    print("Simulation complete")
    return C

# DATA PROCESSING

def downsampleBy(n: int, data:np.ndarray, axis:int=2) -> np.ndarray:
    '''
    downsamples numpy data by n
    axis=0 is rows, axis=1 is columns, axis=2 is both
    '''
    match axis:
        case 0:
            return data[::n, :]
        case 1:
            return data[:, ::n]
        case 2:
            return data[::n, ::n]
        case _:
            return data

def downsampleBy_df(n: int, df:pd.DataFrame) -> pd.DataFrame:
    inds = np.arange(start=0, stop=df.shape[0], step=n)
    downsampled_df = df.iloc[inds, :].copy()
    return downsampled_df

def sim_to_df(
    sim: np.ndarray,
    t: np.ndarray,
    inlet: np.ndarray
) -> pd.DataFrame:
    '''
    converts the numpy data into a unified dataframe
    '''
    print("Saving data...", end="")
    cols = [sec for sec in range(1,sim.shape[1]+1)]
    df = pd.DataFrame(data=sim, columns=cols)
    df.insert(loc=0, column='inlet', value=inlet)
    df.insert(loc=0, column='t', value=t)
    print("Complete")
    return df

# RANDOM

def find_nearest_ind(array:np.ndarray, value):
    return (np.abs(array - value)).argmin()

def get_section_names(data: pd.DataFrame) -> list:
    '''
    extracts the integer columns from a dataframe and returns them in a list
    '''
    sections = []
    for c in data.columns:
        try:
            if (isinstance(int(c), int)): sections.append(c)
        except:
            continue
    return sections