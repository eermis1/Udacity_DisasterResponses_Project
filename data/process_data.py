import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """
    - Takes inputs as two CSV files
    - Merges them into a single dataframe

    Args:
    messages_file_path str: Messages CSV file
    categories_file_path str: Categories CSV file

    Returns:
    merged_df pandas_dataframe: Dataframe obtained from merging the two input data
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")

    return df


def clean_data(df):

    """
    - Cleans the combined dataframe to get ready for use by ML pipeline

    Args:
    df pandas_dataframe: Merged dataframe returned from load_data() function

    Returns:
    df pandas_dataframe: Cleaned data to be used by ML model
    """

    categories = df.categories.str.split(pat=";", expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # binary conversion
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # to handle "2" records in category - related
    categories["related"].replace(to_replace=2, value=1, inplace=True)

    df.drop(columns="categories", inplace=True)
    df = pd.concat([df, categories], axis=1)
    print(df.head())
    df.drop_duplicates(inplace=True)
    print(df.shape)
    return df


def save_data(df, database_filename):

    """
    - Saves cleaned data to an SQL database

    Args:
    df pandas_dataframe: Cleaned data returned from clean_data() function
    database_file_name str: File path of SQL Database into which the cleaned\
    data is to be saved
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('clean_dataset', engine, index=False, if_exists= 'replace')


def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()