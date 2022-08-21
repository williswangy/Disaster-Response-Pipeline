import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download(['wordnet', 'punkt', 'stopwords'])

def load_data(database_filepath):
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table('disaster_response',engine)
    X = df["message"] # message column fro training
    Y = df.iloc[:, 4:] # 36 categories for target
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    # text normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize
    words = word_tokenize(text)
    # stop words removal
    stopword = [w for w in words if w not in stopwords.words("english")]
    # Lemmatization process
    clean_tokens =[WordNetLemmatizer().lemmatize(w) for w in stopword]
    return clean_tokens



def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100]
        #'clf__estimator__min_samples_split': [2, 4]
        } 
    cv = GridSearchCV(pipeline, param_grid=parameters)    
    return cv

    


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
     with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
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