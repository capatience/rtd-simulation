import argparse
import process_files, foobank
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
# sns.set_theme()

def init_argparse():
    # initialize parser for command line arguments
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Plots data that has already been analyzed by main.py",
    )
    
    # add arguments the parser will understand
    # "" is a positional command and is required
    # "-" and "--" are optional (short and long, respectively)
    #  help is the description, type is by default string
    parser.add_argument("-f", "--filename", help="csv data to plot", type=str)

    return parser

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
    fig = plt.figure()
    plt.title("Injection gas in reactor sections")
    for s in sections:
        plt.plot(data.t, data[f"{s}"])
    plt.xlabel("time [s]")
    plt.ylabel("Normalized concentation [1]")
    plt.show()
    return

def plot_sim_color(data:pd.DataFrame):
    ax = plt.subplot()
    cols = foobank.get_section_names(data)
    im = ax.imshow(data[cols], aspect="auto", interpolation='none')

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

def main(args):
    filename = "test_data2.csv"
    direc = "./output/"

    if args.filename:
        filename = args.filename
    
    df = process_files.import_sim_data(direc+filename)
    sections = foobank.get_section_names(data=df)
    plot_sim(data=df, sections=sections)
    # plot_sim_color(df)
    return

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    main(args)