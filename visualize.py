import numpy as np
import pandas as pd
import process_files, foobank
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# filename = "./output/test_data.csv"
# df = process_files.import_sim_data(filename)
# cols = foobank.get_section_names(df)
# xsize = len(cols)

# fig, ax = plt.subplots()
# dataAll = df.loc[:, cols]
# dataDisplay = df.loc[0, cols]

fig = plt.figure()
ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
a=np.random.random((5,5))
im=plt.imshow(a,interpolation='none')

# im, = ax.imshow(dataDisplay, aspect="auto", interpolation='none')

def init():
    im.set_data(np.random.random((5,5)))
    return [im]

def animate(frame):
    # data = dataAll.loc[frame, cols].values
    # im.set_data(np.insert(data, 2))
    a = im.get_array()
    a = a*np.exp(-0.001*frame)
    im.set_array(a)
    return [im]

# ani = FuncAnimation(
#     fig = fig,
#     func = update,
#     frames = df.shape[0],
#     blit = True
# )

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

anim.save('./animations/basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()