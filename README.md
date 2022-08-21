# Disaster-Response-Pipeline
## Table of Contents
1. [Description](#description)
2. [Library](#Library)
3. [Running](#running)


<a name="descripton"></a>
## Description

In this project, an ETL pipeline was built to load the disaster message and categories csv file, clean the data and store the ready-to-use database from the figure eight disaster datasets.
Moreover, a machine learning pipeline was built to train, test and predict the classification results within 36 different categories by using NLTK, pipeline grid search to output a final model.
Finially, a web app has been created to show the classification results from the given messgae input in real time.

<a name="Library"></a>
## Library
* Numpy
* Pandas
* Sciki-Learn
* NLTK*
* SQLalchemy
* Pickle
* Flask
* Plotly

<a name="running"></a>
## Running
To clone the git repository:
```
    git clone https://github.com/williswangy/Disaster-Response-Pipeline.git
```

 To run ETL pipeline to clean data and store the processed data in the database
```
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

 To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file
```
    python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
```
 Run the following command in the app's directory to run your web app.
    `python run.py`

 Go to
 ```
    http://0.0.0.0:3000/
 ```

