import score
import numpy
import os
import requests
import subprocess
import time
import unittest
import joblib
import pytest
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

