import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.utils import setup_logging, save_image

def visualize_numerical_data(df, num_cols):
    """
    Visualizes numerical data by creating summary statistics, histograms,
    and a correlation heatmap.
    """
    # Checking summary statistics
    summary_stats = df[num_cols].describe().T

    # Creating histograms
    df[num_cols].hist(figsize=(14, 14))
    plt.suptitle('Histograms of Numerical Columns')
    hist_figure = plt.gcf()
    save_image(hist_figure, 'numerical_histograms.png')
    plt.show()

    # Plotting the correlation between numerical variables
    plt.figure(figsize=(15, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='0.2f', cmap='YlGnBu')
    plt.title('Correlation Heatmap of Numerical Variables')
    corr_figure = plt.gcf()
    save_image(corr_figure, 'correlation_heatmap.png')
    plt.show()

def visualize_categorical_data(df, cat_cols):
    """
    Visualizes categorical data by printing percentages of sub-categories
    and creating bar plots showing the percentage of attrition.
    """
    for i in cat_cols:
        print(df[i].value_counts(normalize=True))
        print('*' * 40)

    for i in cat_cols:
        if i != 'Attrition':
            plot = (pd.crosstab(df[i], df['Attrition'], normalize='index') * 100).plot(kind='bar', figsize=(8, 4), stacked=True)
            plt.ylabel('Percentage Attrition %')
            plt.title(f'Attrition by {i}')
            bar_figure = plt.gcf()
            save_image(bar_figure, f'{i}_attrition_bar.png')
            plt.show()

def visualize_grouped_means(df, num_cols):
    """
    Visualizes the means of numerical variables grouped by attrition.
    """
    grouped_means = df.groupby(['Attrition'])[num_cols].mean()

    # You can save the grouped means table if needed
    return grouped_means