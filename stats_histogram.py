import matplotlib.pyplot as plt

def plot_histogram(df):
    for column in df.columns:
        if column != 'UID':
            plt.hist(df[column], bins=10, edgecolor='black')
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()