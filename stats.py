import pandas as pd
import matplotlib.pyplot as plt

def descriptive_analysis(df):
    print(df.describe().round(2))

def plot_histogram(df):
    for column in df.columns:
        if column != 'UID':
            plt.hist(df[column], bins=10, edgecolor='black')
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
def plot_scatterplot(df, x_column, y_column):
    plt.scatter(df[x_column], df[y_column])
    plt.title(f'Scatterplot of {y_column} vs {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()
