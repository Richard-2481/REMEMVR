import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def histogram(df, figure):

    if len(figure['xyz']) == 3:

        chart = sns.histplot(df, x=figure['xyz'][0], hue=figure['xyz'][1], bins=figure['xyz'][2], multiple='stack')
        chart.legend(title=figure['labels'][1], labels=[figure['labels'][3], figure['labels'][2]])

        if 'xaxis' in figure:
            # chart.set_xticks(figure['xaxis']['ticks'], labels=figure['xaxis']['labels'])
            # Calculate the bin edges and centers
            # data_min, data_max = df[figure['xyz'][0]].min(), df[figure['xyz'][0]].max()
            data_min, data_max = figure['xaxis']['range'][0], figure['xaxis']['range'][1]
            
            # bin_edges = np.linspace(data_min, data_max, figure['xyz'][2] + 1)
            bin_edges = np.linspace(data_min, data_max, figure['xyz'][2] + 1)
            
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            # Update the x-axis ticks to the bin centers
            chart.set_xticks(bin_centers)
            chart.set_xticklabels(figure['xaxis']['labels'])
    else:
        chart = sns.histplot(df, x=figure['xyz'][0])
        
    chart.set_title(figure['name'])
    chart.set_xlabel(figure['labels'][0])
    chart.set_ylabel('Frequency')

    path = f"plots/{figure['name']} histogram.png"     
    
    plt.savefig(path, bbox_inches='tight')
    plt.clf()

def scatter(df, figure):
        
    if len(figure['xyz']) == 3:
        chart = sns.scatterplot(data=df, x=figure['xyz'][0], y=figure['xyz'][1], hue=figure['xyz'][2])
        sns.regplot(x=figure['xyz'][0], y=figure['xyz'][1], data=df, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'}, scatter=False)

        chart.legend(title=figure['labels'][2], labels=[figure['labels'][4], figure['labels'][3]])
    else:
        chart = sns.scatterplot(data=df, x=figure['xyz'][0], y=figure['xyz'][1])

    chart.set_title(figure['name'])
    chart.set_xlabel(figure['labels'][0])
    chart.set_ylabel(figure['labels'][1])

    path = f"plots/{figure['name']} scatter.png"     
    
    plt.savefig(path, bbox_inches='tight')
    plt.clf()

# def scatter(df, variable, line):
#     if len(variable) == 3:
#         if line == 0:
#             plt.scatter(df[variable[0]], df[variable[1]], c=df[variable[2]], cmap='turbo', alpha=0.5)
#         else:
#             sns.regplot(x=variable[0], y=variable[1], data=df, order=line, line_kws={'color': 'red'})
#         plt.colorbar(label='Sex (0 for Female, 1 for Male)')
#         plt.title(f'Scatter Plot of {variable[0].capitalize()} v {variable[1].capitalize()} Coloured by {variable[2].capitalize()}')
#         path = f'plots/{variable[0]}_{variable[1]}_{variable[2]}_scatter.png'    

#     else:
#         if line == 0:
#             plt.scatter(df[variable[0]], df[variable[1]], alpha=0.5)
#         else:
#             sns.regplot(x=variable[0], y=variable[1], data=df, order=line, line_kws={'color': 'red'})
#         plt.title(f'Scatter Plot of {variable[0].capitalize()} v {variable[1].capitalize()}')
#         path = f'plots/{variable[0]}_{variable[1]}_scatter.png'    

#     plt.xlabel(variable[0].capitalize())
#     plt.ylabel(variable[1].capitalize())
#     plt.savefig(path, bbox_inches='tight')
#     plt.clf()

# def scatter2(df, variable, line, residuals):
#     if residuals == 0:
#         fig, ax = plt.subplots(figsize=(5, 5))
#     else:
#         fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Create two subplots side by side
#         ax = axs[0]

#     if len(variable) == 3:
#         if line == 0:
#             ax.scatter(df[variable[0]], df[variable[1]], c=df[variable[2]], cmap='turbo', alpha=0.5)
#         else:
#             sns.regplot(x=variable[0], y=variable[1], data=df, order=line, ax=ax, line_kws={'color': 'red'})
#         if residuals == 0:
#             fig.colorbar(ax.collections[0], ax=ax, label='Sex (0 for Female, 1 for Male)')
#         path = f'plots/{variable[0]}_{variable[1]}_{variable[2]}_scatter.png'
#     else:
#         if line == 0:
#             ax.scatter(df[variable[0]], df[variable[1]], alpha=0.5)
#         else:
#             sns.regplot(x=variable[0], y=variable[1], data=df, order=line, ax=ax, line_kws={'color': 'red'})
#         path = f'plots/{variable[0]}_{variable[1]}_scatter.png'

#     ax.set_title(f'Scatter Plot of {variable[0].capitalize()} v {variable[1].capitalize()}')
#     ax.set_xlabel(variable[0].capitalize())
#     ax.set_ylabel(variable[1].capitalize())

#     if residuals == 1:
#         ax = axs[1]
#         sns.residplot(x=variable[0], y=variable[1], data=df, order=line, ax=ax, scatter_kws={'alpha': 0.5})
#         ax.set_title(f'Residual Plot of {variable[0].capitalize()} v {variable[1].capitalize()}')
#         ax.set_xlabel(variable[0].capitalize())
#         ax.set_ylabel('Residuals')

#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight')
#     plt.clf()

def violin(df):
    for column in df.columns:
        if column != 'UID':
            sns.violinplot(x=column, data=df)
            plt.title(f'Violin plot of {column}')
            plt.xlabel(column)
            plt.ylabel('Value')
            path = f'plots/{column}_violin.png'
            plt.savefig(path, bbox_inches='tight')
            plt.clf()

def box(df):
    for column in df.columns:
        if column != 'UID':
            sns.boxplot(x=column, data=df)
            plt.title(f'Boxplot of {column}')
            plt.xlabel(column)
            plt.ylabel('Value')
            path = f'plots/{column}_boxplot.png'
            
            # Label outliers with UID values if 'UID' column exists
            if 'UID' in df.columns:
                outliers = df[df[column].apply(lambda x: x < df[column].quantile(0.25) - 1.5 * (df[column].quantile(0.75) - df[column].quantile(0.25))) | df[column].apply(lambda x: x > df[column].quantile(0.75) + 1.5 * (df[column].quantile(0.75) - df[column].quantile(0.25)))]
                for index, row in outliers.iterrows():
                    plt.text(row.name, row[column], row['UID'], horizontalalignment='center', verticalalignment='bottom', color='red')
            
            plt.savefig(path, bbox_inches='tight')
            plt.clf()



def plot(df, figure):
   
    if figure['plot_type'] == 'histogram':
        histogram(df, figure)

    if figure['plot_type'] == 'scatter':
        scatter(df, figure)

    if figure['plot_type'] == 'violin':
        violin(df, figure)

    if figure['plot_type'] == 'box':
        box(df, figure)