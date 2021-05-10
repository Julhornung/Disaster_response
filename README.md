# Disaster Response Pipeline Project

Project as a part of the Udacity Data Scientist Nano Degree Program.

This package will be used to analyse text messages from a disaster response scenario provided by Figure Eight. The aim is to classifiy the incoming messages
in such a way that the coordination of different organizations involved in disaster response is facilitated. 
A fast, efficient and reliable classification of a huge number of different text messages e.g. after a natural desaster will be very
helpful to coordinate the supply of medical help, water and food and the assignment of rescue teams to places where the respectiv ressource
is neede most. 

# Installation:

Required libraries:

nltk 3.3.0
numpy 1.15.2
pandas 0.23.4
scikit-learn 0.20.0
sqlalchemy 1.2.12
pickle 4.0

# Files:


workspace/data/process_data.py:

	A data cleaning pipeline that:
		Loads the messages and categories datasets
        	Cleans data
		Merges the two datasets
		Stores it in a SQLite database
        
workspace/model/train_classifier.py:

	A machine learning pipeline that:
		Loads data from the SQLite database
		Splits the dataset into training and test sets
		Builds a text processing and machine learning pipeline
		Trains and tunes a model using GridSearchCV
		Outputs results on the test set
		Exports the final model as a pickle file

app/run.py:
		
	A flask app used to deploy a web page:
		Visualizations of data
		Classification of incoming text messages	



### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Acknowledgements

I wish to thank Figure Eight for the dataset and the very interesting project. Furthermore I want to thank Udacity for the nice and interesting course, project and the helpful comments to my projects. 
