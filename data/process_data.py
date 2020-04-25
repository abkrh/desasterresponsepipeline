import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads source data from csv files and creates and returns a single  dataframe
    
    INPUTS:
    messages_filepath - path to the csv file for messages
    categories_filepath - path to the csv file for categories
    
    OUTPUTS:
    df - the created dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')   
    return df
    
     
    

def clean_data(df):
    '''
    Cleans the datframe by expanding the categories into indivudal columns and removes duplicates 
    
    INPUTS:
    df - the raw dataframe continaing unprocessed contents of the messages and categories 
    
    OUTPUTS:
    df - the cleaned dataframe
    '''
    categories = df['categories'].str.split(pat=';', expand=True)
    
    category_colnames = list(map(lambda f : f[:-2], categories.loc[0]))
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string and convert to int
        categories[column] = categories[column].str[-1].astype(int)

    
    df.drop("categories",inplace=True, axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df[df['related'] != 2]
    df.drop_duplicates(inplace=True)
    #df.replace({"related":2}, 0, inplace=True)
    #df.replace({np.nan: 0}, inplace=True)
    return df


def save_data(df, database_filename):
    '''
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disasters', engine, index=False)  


def main():
    '''
    main entry point fot the file, runs all other functions to procees the input files and export the database
    
    INPUTS:
    No direct inputs, but takes the sys.argv, these are:
    messages_filepath - the filepath tot the csv file containting the messages
    categories_filepath - the filepath tot the csv file containting the categories
    database_filepath -  - the filepath  to save the database created
    
    OUTPUTS:
    None.    
    '''
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