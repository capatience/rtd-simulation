import process_files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def lineplot(Xs, t):
    for X in Xs:
        plt.plot(t, X)
    return

def plot_sim(data: pd.DataFrame, sections: list=[1]):
    '''
    plots a line of the concentration over time for a list of sections (list[int])
    '''
    for s in sections:
        plt.plot(data.t, data[f"{s}"])
    plt.show()
    return

def extract_section_names(data: pd.DataFrame) -> list:
    sections = []
    for c in data.columns:
        try:
            sections.append(int(c))
        except:
            continue
    return sections

def main():
    filename = "./output/test_data2.csv"
    df = process_files.import_sim_data(filename)
    sections = extract_section_names(data=df)
    plot_sim(data=df, sections=sections)
    return

if __name__ == "__main__":
    main()