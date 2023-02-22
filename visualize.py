import numpy as np
import pandas as pd
import process_files, foobank
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

filename = "./output/test_data.csv"
df = process_files.import_sim_data(filename)
cols = foobank.get_section_names(df)
xsize = len(cols)

fig, ax = plt.subplots()
dataAll = df.loc[:, cols]
dataDisplay = df.loc[0, cols]
ln, = ax.imshow(dataDisplay, aspect="auto")

def update(frame):
    data = dataAll.loc[frame, cols].values
    ln.set_data(np.insert(data, 2))
    return ln,

ani = FuncAnimation(
    fig = fig,
    func = update,
    frames = df.shape[0],
    blit = True
)

plt.show()