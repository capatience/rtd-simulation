import numpy as np
import pandas as pd
import process_files, foobank
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

INPUT_FILENAME = "./output/test_data2.csv"

def listRangeInclusive(beg: int, end: int) -> list[int]:
    '''
    Creates a list of integers from beg to end (inclusive)
    '''
    return np.arange(beg, end+1).tolist()

def listIntToString(lst: list[int]) -> list[str]:
    '''
    Converts a list of int to list of strings
    '''
    return [str(i) for i in lst]

def rangeCols(beg: int, end: int) -> list[str]:
    '''
    Creates a list of an ascending range of integers as strings
    '''
    return listIntToString(listRangeInclusive(beg, end))

def getDataDisplayed(data: np.ndarray, row: int) -> np.ndarray:
    dataDisplayed = np.ones([1, data.shape[1]]) * data[row]
    return dataDisplayed

# GRAPH VISUALS
parameters = process_files.csv_to_dict("./parameters.csv")
v1size = parameters['Nz_CPOX_UPPER']
v2size = parameters['Nz_CPOX_LOWER']
v3size = parameters['Nz_FT']
shapes = {
    'CPOX_UPPER_COLS': rangeCols(1,v1size),
    'CPOX_LOWER_COLS': rangeCols(v1size+1,v1size+v2size),
    'FT_COLS': rangeCols(v1size+v2size+1,v1size+v2size+v3size),
}

# GET DATA
df = process_files.import_sim_data(INPUT_FILENAME)
cols = foobank.get_section_names(df)
dataAll = df.loc[:, cols].values
dataCPOXUPPER = df.loc[:, shapes['CPOX_UPPER_COLS']].values
dataCPOXLOWER = df.loc[:, shapes['CPOX_LOWER_COLS']].values
dataFT = df.loc[:, shapes['FT_COLS']].values
timeOffset = df.t.iloc[0]
timeFinal = df.t.iloc[-1] - timeOffset

# plotting rate
fastforward = 10
totalframes = dataAll.shape[0]
frames = int(totalframes / fastforward)
fps = 120 # frames per second
spf = 0.01 # seconds per frame


# FIGURE
gridspec_kw = {
    'width_ratios': [1,1,4],
    'height_ratios': [2]
}
fig = plt.figure()
spec = gridspec.GridSpec(ncols=3, nrows=2, width_ratios=[1,1,2], height_ratios=[1,3])

# plt.subplot(1,3,1)
# im = plt.imshow(
#     X = getDataDisplayed(data=dataAll, row=0),
#     interpolation='none',
#     aspect = 'auto',
#     vmin = 0,
#     vmax = 1
# )

ax1 = fig.add_subplot(spec[0,0])
imCPOXUPPER = plt.imshow(
    X = getDataDisplayed(data=dataCPOXUPPER, row=0),
    interpolation='none',
    aspect = 'auto',
    vmin = 0,
    vmax = 1
)
# plt.axes("off")

ax2 = fig.add_subplot(spec[0,1])
imCPOXLOWER = plt.imshow(
    X = getDataDisplayed(data=dataCPOXLOWER, row=0),
    interpolation='none',
    aspect = 'auto',
    vmin = 0,
    vmax = 1
)
# plt.axes("off")

ax3 = fig.add_subplot(spec[:,2])
imFT = plt.imshow(
    X = getDataDisplayed(data=dataFT, row=0),
    interpolation='none',
    aspect = 'auto',
    vmin = 0,
    vmax = 1
)
# plt.axes("off")

def init():
    imCPOXUPPER.set_data(getDataDisplayed(data=dataCPOXUPPER, row=0))
    imCPOXLOWER.set_data(getDataDisplayed(data=dataCPOXLOWER, row=0))
    imFT.set_data(getDataDisplayed(data=dataFT, row=0))
    return [imCPOXUPPER, imCPOXLOWER, imFT]

def animate(frame):
    progress = frame/frames*100
    if (progress%5 == 0): print(f"Animating...{progress:.0f}%")
    plt.title(f"{(df.t.iloc[frame]-timeOffset):.1f}/{timeFinal:.1f}") # 
    
    imCPOXUPPER.set_data(getDataDisplayed(data=dataCPOXUPPER, row=frame*fastforward))
    imCPOXLOWER.set_data(getDataDisplayed(data=dataCPOXLOWER, row=frame*fastforward))
    imFT.set_data(getDataDisplayed(data=dataFT, row=frame*fastforward))
    return [imCPOXUPPER, imCPOXLOWER, imFT]

def save_animation(filename: str, fps: int, save: bool):
    if (save): 
        print("Saving", end="...")
        anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])
        print("Complete")

anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=1, blit=True, repeat=False)

plt.show()
# save_animation(filename='./animations/basic_animation.mp4', fps=fps, save=True)