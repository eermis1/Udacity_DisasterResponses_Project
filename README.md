# Udacity Data Scientist Nanodegree Program 
## Disaster Response Pipeline Project
***

## Description

This repository has been created for Udacity Data Scientist Nanodegree Program - Data Engineering Part - Disaster Response Pipeline Project.
The dataset has been provided by Figure Eight and it contains pre-labelled tweet and messages from real-life disaster 
The aim of the project is to build a NLP Machine Learning Pipeline to categorize emergency messages based on the needs communicated by sender.
The predictions from the pipeline will be used by organizations via web app that has been designed in the project.

The project is consisted of 3 main parts.

1. **ETL Pipeline**: Extract data from source, clean and save into a SQLite DB.
2. **Machine Learning Pipeline**: To train the model in order to be able to classify the messages correctly.
3. **Flask & Plotly Based Web App**: Interactive web app that allows users to enter message and get classification predictions.

## Getting Started

### Directory Structure
~~~~~~~
        Udacity_DisasterResponses_Project
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- CleanDataDB.db
                |-- process_data.py
          |-- models
                |-- model.pkl
                |-- train_classifier.py
          |-- Jupyter_Notebooks
                |-- ETL Pipeline Preparation.ipynb
                |-- ETL Pipeline Preparation.html
                |-- ML Pipeline Preparation.ipynb
                |-- ML Pipeline Preparation.html                
          |-- README
~~~~~~~

### Dependencies
* Python 3.5+ (I used Python 3.7)
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

### Installation & Instructions


1. Download the repository <br>
  `git clone https://github.com/eermis1/Udacity_DisasterResponses_Project.git`

2. Run the following commands in the project's root directory to set up your database and machine learning model.

    - To run ETL pipeline by adding required filepaths. See an example below <br>
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline by adding required filepaths. See an example below <br>
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app. <br>
     `python run.py`

4. Go to http://0.0.0.0:3001/


### Author

The repository has been created by ***Evren Ermiş*** <br>
- [Github](https://github.com/eermis1)
- [Linkedin](www.linkedin.com/in/evrenermis92)
- [Kaggle](https://www.kaggle.com/evrenermis/)

### Licanse

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
