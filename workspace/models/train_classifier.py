import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    
    engine = create_engine('sqlite:///' + database_filepath)
    
    df = pd.read_sql_table('Table', 'sqlite:///' + database_filepath)
    
    df = df.dropna()
    
    X = df['message'].values
    
    Y = df.drop(['message', 'original', 'id'], axis = 1).values

    category_names = df.drop(['message', 'original', 'id'], axis = 1).columns
    
    #for column in Y_df.columns:
        
       #print(column)
       #print(df[column].unique())
    
    return(X, Y, category_names)


def tokenize(text):
    
    porter = PorterStemmer()
    
    text = text.lower()
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    words = nltk.word_tokenize(text)
    
    words = [w for w in words if w not in stopwords.words("english")]
    
    words = [porter.stem(w) for w in words]
    
    pass


def build_model():
    
    model = Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(KNeighborsClassifier()))
                ])
    
    return(model)


def evaluate_model(model, X_test, y_test, category_names):
    
    y_pred = model.predict(X_test)
    
    count = 0
    
    for category in category_names: 
            print(classification_report(np.hstack(y_test[:,count]),np.hstack(y_pred[:,count])))
    count += 1
    
    return()


def save_model(model, model_filepath):
    
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    return()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
               
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(np.unique(X_train))
        print(np.unique(X_test))
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()