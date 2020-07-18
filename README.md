# Udacity Data Scientist Nanodegree Program 
## Disaster Response Pipeline Project
***

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
                |-- visualizations.py                
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

### Installation & Instructions

1. Create virtual environment and activate it <br>
   `python3 -m venv env` <br>
   `source env/bin/activate` <br>

2. Download the repository to virtual environment <br>
   `cd env` <br>
   `git clone https://github.com/eermis1/Udacity_DisasterResponses_Project.git` <br>
   `cd Udacity_DisasterResponses_Project` <br>

3. Install required libraries <br>
   `pip install numpy`<br>
   `pip install scipy` <br>
   `pip install pandas` <br>
   `pip install sklearn` <br>
   `pip install nltk` <br>
   `pip install SQLalchemy` <br>
   `pip install flask` <br>
   `pip install plotly` <br>
   
4. Go to app directory <br>
   `cd app`
   
5. Run "run.py" <br>
   `python run.py` <br>

6. Go to http://0.0.0.0:3001/ <br>


### Author

The repository has been created by ***Evren Ermiş*** <br>

- [Linkedin](www.linkedin.com/in/evrenermis92)
- [Github](https://github.com/eermis1)
- [Kaggle](https://www.kaggle.com/evrenermis/)
