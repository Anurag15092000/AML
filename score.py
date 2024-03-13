import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report ,ConfusionMatrixDisplay
import joblib
import click
import warnings

warnings.filterwarnings("ignore")
best_model= joblib.load('model_pipeline.pkl')


# @click.command()
# @click.option('--text',prompt='Input text',
#               help='Input text to be classified')
# @click.option('--threshold',prompt='threshold for scoring')

def score(text:str, model, threshold):


    propensity = model.predict_proba([text]).tolist()[0][1]
    if propensity >= threshold:
        prediction = 1
    else:
        prediction = 0
    return prediction, propensity

# if __name__=='__main__':
#     print(score())
