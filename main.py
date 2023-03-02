import pandas as pd
import numpy as np
import process_files, foobank
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()

CSVFILEPATH = "./parameters.csv"
# MAHDI_DATA = "./data/data-mahdi/400C_15bar_12lpm_73nlpm_repeat.csv"
OUTPUT_FILENAME = "test_data2"

def main():
    # load the parameters (set by the user)
    parameters = process_files.csv_to_dict(CSVFILEPATH)

    # select the dataset you want to load (not necessary for simulation)
    # data = process_files.import_data_mahdi(MAHDI_DATA)


    # PARAMETERS

    # time
    tstart = parameters['tstart'] # time of injection [s]
    tstop = parameters['tstop'] # time when the injection stops (used for simulation) [s]
    duration_injection = parameters['dur_injection']
    tinjection = parameters['tinjection']

    # inlet (0)
    Q0 = parameters['Q'] /1000 /60  # volumetric flowrate at the inlet [m^3/s]
    
    # section CPOX UPPER (1)
    D1 = parameters['D_CPOX_UPPER'] * 2.54 / 100 # diameter [m]
    L1 = parameters['L_CPOX_UPPER'] * 2.54 / 100 # length [m]
    Nz1 = parameters['Nz_CPOX_UPPER'] # number of sections []
    dV1 = np.ones(Nz1,)*foobank.calc_dV(D1, L1, Nz1) # differential volume, constant [m3]
    Q1 = Q0 * np.ones_like(dV1) # flowrate [m^3/s] todo: update this to reflect temperature Q1 = Q0 * T1/T0
    
    # section CPOX LOWER (2)
    D2 = parameters['D_CPOX_LOWER'] * 2.54 / 100
    L2 = parameters['L_CPOX_LOWER'] * 2.54 / 100
    Nz2 = parameters['Nz_CPOX_LOWER']
    dV2 = np.ones(Nz2,)*foobank.calc_dV(D2, L2, Nz2)
    Q2 = Q0 * np.ones_like(dV2)

    # section FT (3)
    D3 = parameters['D_FT'] * 2.54 / 100
    L3 = parameters['L_FT']# * 2.54 / 100
    Nz3 = parameters['Nz_FT']
    dV3 = np.ones(Nz3,)*foobank.calc_dV(D3, L3, Nz3)
    Q3 = Q0 * np.ones_like(dV3)

    # data w.r.t. TIME
    timestep = parameters['dt'] # timestep
    t = np.arange(tstart,tstop,timestep)
    tinject_start_index = foobank.find_nearest_ind(t, tinjection)
    tinject_end_index = foobank.find_nearest_ind(t, tinjection + duration_injection)
    inlet = np.zeros_like(t) # assumes a constant input of 1
    inlet[tinject_start_index: tinject_end_index] = 1
    IC = 0 # starts empty

    # data w.r.t. SECTION
    dV = np.concatenate([dV1, dV2, dV3])
    Q = np.concatenate([Q1, Q2, Q3])

    sim = foobank.propagate(
        t = t,
        inlet = inlet,
        dV = dV,
        Q = Q,
        IC = IC,
        verbose=True
    )

    sim_downsample_cols = downsampleBy(n=1, data=sim, axis=1)

    dfsim = sim_to_df(sim=sim_downsample_cols, t=t, inlet=inlet)
    dfsim_downsample_time = downsampleBy_df(n=4, df=dfsim)

    process_files.export_df_data(dfsim_downsample_time, OUTPUT_FILENAME)
    return

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

if __name__ == "__main__":
    main()