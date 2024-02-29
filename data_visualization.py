import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_fetch import fetch_data


def save_image(fig, filename):
    if not os.path.exists('images'):
        os.makedirs('images')
    filepath = os.path.join('images', filename)
    fig.savefig(filepath)


def print_info(df):
    pd.set_option('display.max_columns', None)
    print('Dataframe shape:', df.shape, '\n')
    print('Dataframe sample: \n', df.sample(5), '\n')
    print('Dataframe types: \n', df.dtypes, '\n')
    print('Dataframe describe: \n', df.describe().T, '\n')
    print('Dataframe duplicates: \n', df.duplicated().sum(), '\n')
    print('Missing values: \n', df.isnull().sum(), '\n')


def plot_correlation_matrix(df):
    plt.figure(figsize=(18, 8))
    sns.heatmap(df.corr(), annot=True, cmap='crest', fmt='.2f', linewidths=2)
    plt.title('Correlation Matrix', fontsize=20)
    save_image(plt, 'correlation_matrix.png')


def plot_distribution(df):
    columns = df.columns[:-1]
    plt.figure(figsize=(20, 10))
    plt.suptitle('Density distribution of features', fontsize=20)

    for i, column in enumerate(columns, start=1):
        plt.subplot(3, 5, i)
        sns.histplot(x=column, hue='class', data=df, palette='crest', kde=True)

    save_image(plt, 'distribution')


def plot_pairplot(df):
    pairplot = sns.pairplot(df, hue='class', palette='viridis')
    save_image(pairplot, 'pairplot.png')

def detect_outliers(df):
    columns = df.columns[:-1]
    for col_name, col_values in df[columns].items():
        q1 = col_values.quantile(0.25)
        q3 = col_values.quantile(0.75)
        irq = q3 - q1
        outliers = col_values[(col_values <= q1 - 1.5 * irq) | (col_values >= q3 + 1.5 * irq)]
        percentage_outliers = len(outliers) * 100.0 / len(df)
        print("Column {} outliers = {}%".format(col_name, round(percentage_outliers, 2)))



def plot(df):
    plot_correlation_matrix(df)
    plot_distribution(df)
    plt.show()
    plot_pairplot(df)
    plt.tight_layout()


def main():
    try:
        df = pd.read_csv('wine.csv')
    except FileNotFoundError:
        fetch_data()
        try:
            df = pd.read_csv('wine.csv')
        except FileNotFoundError:
            print('Error fetching data. Please try again.')
            exit(1)

    print_info(df)
    #plot(df)
    detect_outliers(df)


if __name__ == '__main__':
    main()
