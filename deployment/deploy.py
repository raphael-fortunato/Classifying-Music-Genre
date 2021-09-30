import torch
import os
import sys

from bentoservice import GenreClassification

# IN order to get our module higher up
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
from model import AudioModel
from main import get_args
sys.path.pop(0)

if __name__ == '__main__':
    # Set the path of our model and load the model
    path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models', 'deployment_model.pt'))

    model = toch.load(path)
    model.eval()

    # Create a iris classifier service instance
    genre_classifier = GenreClassification()

    # Pack the newly trained model artifact
    genre_classifier.pack('model', model)

    # Save the prediction service to disk for model serving
    saved_path = genre_classifier.save()
    print('packaged model saved at: ', saved_path)

