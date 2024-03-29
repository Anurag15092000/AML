import score
import numpy
import os
import random
import requests
import subprocess
import time
import unittest
import joblib
import pytest
import pandas as pd
from score import score

import warnings

warnings.filterwarnings("ignore")
best_model= joblib.load('model_pipeline.pkl')
# threshold=0.7

# label,prop=score.score(sent,model,threshold)




def test_smoke():
  threshold=0.5
  txt='Win a million dollars'
  pred,prop = (None,None)
  pred,prop = score(txt, best_model, threshold)
  assert pred is not None,"Variable should not be None"
  assert prop is not None,"Variable should not be None"


def test_format():
    threshold=0.5
    txt='Win a million dollars'
    pred,prop = score(txt, best_model, threshold)
    assert isinstance(pred, int)
    assert isinstance(prop, float)



def test_threshold_0():
    threshold=0
    txt='Win a million dollars'
    pred,prop=score(txt,best_model,threshold)
    assert pred==1


def test_threshold_1():
    threshold=1
    txt='Win a million dollars'
    pred,prop=score(txt,best_model,threshold)
    assert pred==0

def test_spam():
    threshold=0.5
    txt='Win a million dollars'
    label,prop=score(txt,best_model,threshold)
    assert label == 1

def test_ham():
    threshold=0.5
    txt='Would love to connect with you'
    label,prop=score(txt,best_model,threshold)
    assert label == 0

class TestDocker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the Docker image
        subprocess.run(["docker", "build", "-t", "assignment_4", "."], check=True)
        # Run the Docker container
        cls.container_id = subprocess.check_output(
            ["docker", "run", "-d", "-p", "5000:80", "assignment_4"]
        ).decode("utf-8").strip()
        # Load input texts from the CSV file
        cls.test_df = pd.read_csv(r'/workspaces/AML/test_dataset.csv')
        time.sleep(5)  # Wait for the server to start

    def test_docker(self):
        r = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[r, 0]
        # Test the /score endpoint
        test_data = {'text': text}
        response = requests.post('http://127.0.0.1:5000/score', json=test_data)
        self.assertEqual(response.status_code, 200)
        # You may want to add more assertions here based on your expected response

    @classmethod
    def tearDownClass(cls):
        # Stop and remove the Docker container
        subprocess.run(["docker", "stop", cls.container_id], check=True)
        subprocess.run(["docker", "rm", cls.container_id], check=True)        
if __name__ == '__main__':
    unittest.main()