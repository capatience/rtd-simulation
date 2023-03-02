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

def getDataDisplayed(data: np.ndarray, row: int, thickness: int) -> np.ndarray:
    dataDisplayed = np.ones([thickness, data.shape[1]]) * data[row]
    return dataDisplayed

# GRAPH VISUALS
shapes = {
    'CPOX_UPPER_COLS': rangeCols(1,20),
    'CPOX_LOWER_COLS': rangeCols(21,40),
    'FT_COLS': rangeCols(41,60),
    'CPOX_LOWER_VOLUME': 10,
    'CPOX_UPPER_VOLUME': 30,
    'FT_VOLUME': 100
}
baseThickness = 2
upperThickness = baseThickness# * shapes['CPOX_UPPER_VOLUME']
lowerThickness = baseThickness# * shapes['CPOX_LOWER_VOLUME']
ftThickness = baseThickness# * shapes['FT_VOLUME']

# GET DATA
df = process_files.import_sim_data(INPUT_FILENAME)
cols = foobank.get_section_names(df)
dataAll = df.loc[:, cols].values
dataCPOXUPPER = df.loc[:, shapes['CPOX_UPPER_COLS']].values
dataCPOXLOWER = df.loc[:, shapes['CPOX_LOWER_COLS']].values
dataFT = df.loc[:, shapes['FT_COLS']].values
frames = dataAll.shape[0]
fps = 120 # frames per second
spf = 0.01 # seconds per frame
timeOffset = df.t.iloc[0]
timeFinal = df.t.iloc[-1] - timeOffset

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
    X = getDataDisplayed(data=dataCPOXUPPER, row=0, thickness=upperThickness),
    interpolation='none',
    aspect = 'auto',
    vmin = 0,
    vmax = 1
)

ax2 = fig.add_subplot(spec[0,1])
imCPOXLOWER = plt.imshow(
    X = getDataDisplayed(data=dataCPOXLOWER, row=0, thickness=lowerThickness),
    interpolation='none',
    aspect = 'auto',
    vmin = 0,
    vmax = 1
)

ax3 = fig.add_subplot(spec[:,2])
imFT = plt.imshow(
    X = getDataDisplayed(data=dataFT, row=0, thickness=ftThickness),
    interpolation='none',
    aspect = 'auto',
    vmin = 0,
    vmax = 1
)

def init():
    imCPOXUPPER.set_data(getDataDisplayed(data=dataCPOXUPPER, row=0, thickness=upperThickness))
    imCPOXLOWER.set_data(getDataDisplayed(data=dataCPOXLOWER, row=0, thickness=lowerThickness))
    imFT.set_data(getDataDisplayed(data=dataFT, row=0, thickness=ftThickness))
    return [imCPOXUPPER, imCPOXLOWER, imFT]

def animate(frame):
    print(f"{frame/frames*100:.0f}%")
    plt.title(f"{(df.t.iloc[frame]-timeOffset):.1f}/{timeFinal:.1f}") # 
    
    imCPOXUPPER.set_data(getDataDisplayed(data=dataCPOXUPPER, row=frame, thickness=upperThickness))
    imCPOXLOWER.set_data(getDataDisplayed(data=dataCPOXLOWER, row=frame, thickness=lowerThickness))
    imFT.set_data(getDataDisplayed(data=dataFT, row=frame, thickness=ftThickness))
    return [imCPOXUPPER, imCPOXLOWER, imFT]

def save_animation(filename: str, fps: int, save: bool):
    if (save): 
        print("Saving", end="...")
        anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])
        print("Complete")

anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=1, blit=True, repeat=False)

# plt.show()
save_animation(filename='./animations/basic_animation.mp4', fps=fps, save=True)