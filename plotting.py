import process_files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
# sns.set_theme()

def lineplot(Xs, t):
    '''
    simple overlaid line plot of multiple data
    '''
    for X in Xs:
        plt.plot(t, X)
    return

def plot_sim(data: pd.DataFrame, sections: list=[1]):
    '''
    plots a line of the concentration over time (column: t) for a list of sections (list[int])
    Each line is from a column named as an integer
    '''
    for s in sections:
        plt.plot(data.t, data[f"{s}"])
    plt.show()
    return

def extract_section_names(data: pd.DataFrame) -> list:
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

def plot_sim_color(data:pd.DataFrame):
    ax = plt.subplot()
    cols = extract_section_names(data)
    im = ax.imshow(data[cols], aspect="auto")

    # changing the y-labels
    tbeg = 0
    tend = data.t.max()-data.t.min()
    labels = np.arange(tbeg, tend, 5)
    scale = int(data.shape[0] / tend)
    ticks = [scale * label for label in labels] # where they're actually placed
    plt.yticks(ticks=ticks,labels=labels)

    plt.xlabel("Volumetric sections")
    plt.ylabel("Time [s]")

    # create an Axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.ylabel("Normalized concentration")

    plt.show()
    return

def main():
    filename = "./output/test_data2.csv"
    df = process_files.import_sim_data(filename)
    sections = extract_section_names(data=df)
    # plot_sim(data=df, sections=sections)
    plot_sim_color(df)
    return

if __name__ == "__main__":
    main()