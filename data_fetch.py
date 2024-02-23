import pandas as pd
from ucimlrepo import fetch_ucirepo

"""
Aeberhard,Stefan and Forina,M.. (1991). 
Wine. UCI Machine Learning Repository. 
https://doi.org/10.24432/C5PC7J.
"""


def fetch_data():
    try:
        data = fetch_ucirepo(id=109)
        X = data.data.features
        y = data.data.targets
        data = pd.concat([X, y], axis=1)
        data.to_csv('wine.csv', index=False)
        print('Data successfully saved as wine.csv.')

    except Exception as e:
        print(f'Error fetching data: {e}')


def main():
    fetch_data()


if __name__ == '__main__':
    main()
