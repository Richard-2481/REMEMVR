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
import numpy as np
import seaborn as sns

def plot_scatterplot(df, x_column, y_column):
    sns.scatterplot(x=df[x_column], y=df[y_column], hue=df['education'])
    plt.title(f'Scatterplot of {y_column} vs {x_column} with Regression Line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()
