import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import nltk
import pickle
nltk.download(['punkt', 'wordnet'])
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    '''
    Laods the main data base
    
    INPUTS:
    database_filepath - filepath to the source database
    
    OUTPUTS:
    X - the source messages
    Y - the source labels
    category_names - the category names
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Disasters', con=engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    '''
    tokenizes the text
    
    INPUTS:
    text - the text to be tokenized
    
    OUTPUTS:
    clean_tokens - tokenized and cleaned tokens (all in lower case)
    '''
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds the RandomForestClassifier classifier model
    
    INPUTS:
    None
    
    OUTPUTS:
    cv - the GridSearchCV model
    '''
    print("Building model....", end="")
    moc = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
        ])
    
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10],
              'clf__estimator__criterion': ['gini', 'entropy'],
              'clf__estimator__min_samples_leaf':[1, 5, 10]}
    
 

    cv = GridSearchCV(pipeline, parameters, verbose=1)#, cv=3, n_jobs=-1)
    print("...Model Completed!")
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the models performance
    
    INPUTS:
    model - the model to evaluate
    X_test - the test set
    Y_test - the label for the test set
    category_names - the category names
    
    OUTPUTS:
    None    
    '''
    y_pred = model.predict(X_test)
    
    # Print the classification report and accuracy score
    print(classification_report(Y_test, y_pred, target_names=category_names))
    print('---------------------------------')
    print('----------MODEL EVALUATION-------')
    print('---------------------------------')
    for i in range(Y_test.shape[1]):
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:,i])))



def save_model(model, model_filepath):
    '''
    Saves the model
    
    INPUTS:
    model - the model to save
    model_filepath - the filepath to save the model
    
    OUTPUTS:
    None
    
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    main entry point fot the file, runs all other functions to procees the input files and export the database
    
    INPUTS:
    No direct inputs, but takes the sys.argv, these are:
    database_filepath - the filepath to the source database
    model_filepath -  - the filepath  to save the model created
    
    OUTPUTS:
    None.    
    '''
        
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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