import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
      Function:
      merge the csv files into one dataframe 
      Args:
      messages_filepath (str): the file path of messages csv file
      categories_filepath (str): the file path of categories csv file
      Returns:
      df (DataFrame): A dataframe of a combination of the csv files.
      """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    """
      Function:
      Expand the categories for the df dataframe from the row level into the colomen as features
      Args:
      df (DataFrame): A dataframe of disaster dataframe need to be cleaned
      Returns:
      df (DataFrame): A cleaned version of dataframe is ready to save to the database
    """
    # Split `categories` into separate category columns.
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = list(categories.iloc[:1].applymap(lambda x: x[:-2]).iloc[0,:])
    # Rename the columns
    categories.columns = category_colnames
    # Convert category values to 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop(["categories"],axis = 1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1, join = 'inner' )
    # Remove the value 2 for related field since it is mean
    df.drop_duplicates(inplace=True)

    df = df[df['related'] != 2]
    return df


def save_data(df, database_filename):
    """
       Function:
       Save the Dataframe df in a database
       Args:
       df (DataFrame): A dataframe of messages and categories
       database_filename (str): The file name of the database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response', engine, index=False,if_exists='replace')


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