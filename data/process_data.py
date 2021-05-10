import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(file1, file2):
    """
    Description: This function reads in two .csv files as pd.DataFrame object. These dataframes will be merged.

    Arguments:
        file1:                  file path to .csv file 1; needs 'id' column
        file2:                  file path to .csv file 2; needs 'id' column
    
    Returns:
        df:                     merged dataframe of both .csv files. merged on "id" column
        df2:                    dataframe object of .csv file 2
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.merge(df1, df2, on=["id"])
    
    return(df, df2)


def clean_data(df, categories):
    """
    used to clean the two dataframes.
    The categories df contains the column 'categories'. 
    Each row in this column represents a list of "categories" with an associated boolean 1 or 0.
    This function splits this list, cleans the entries and represents each "category" in a separate
    column.
    df will be used to remerge the final categories dataframe to df.

    Arguments:
        df:                 Dataframe object 
        categories:         Dataframe object
    
    Returns:
        df:                 merged and cleaned dataframe
    """
    
    
    categories = categories['categories'].str.split(';').apply(pd.Series)
    
    row = categories.iloc[0].values.flatten().tolist()
    category_colnames = [i.split('-', 1)[0] for i in row]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str.strip().str[-1]
        categories[column] = pd.to_numeric(categories[column])
        
    categories['related'] = categories['related'].replace(2,1)
    
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1)
    df = df.drop('genre', axis = 1)
    df = df.drop_duplicates().dropna()
    
    for column in df.columns:
        print(column)
        print(df[column].unique())
      
    return(df)


def save_data(df, database_filename):
    """
    will load the transformed dataframe to a SQL database.
    
    Arguments:
        df:                        Dataframe object 
        database_filename:         Dataframe object
    
    Returns:
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Table', engine, index=False, if_exists = 'replace')

    pass  


def main():
    """
    combines the different functions, to load, clean and safe data to a database

    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
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