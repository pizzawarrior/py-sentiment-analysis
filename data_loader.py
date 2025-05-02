import pandas as pd


# link to dataset, with information: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

data_file = 'data.csv'

def read_data(data_file):
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError as e:
        print(f'Error importing data: {e}')
    print('Data loaded successfully!')
    return df
