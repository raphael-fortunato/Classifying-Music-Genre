import os
import json
import shutil
import numpy as np
import math
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import skimage.io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Get the genres from the dataset
FOLDER = "./dataset/gtzan/genres_original"
GENRES      = [name for name in os.listdir(FOLDER) if os.path.isdir(os.path.join(FOLDER, name))]
NUM_SEGMENTS = 5
SAMPLE_RATE = 22050
TRACK_DUR = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DUR
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
HOP_LENGTH = 512
N_FTT = 2048
NUM_MFCC = 13



# Make a new directory where we store the new files
def CreateNewRepositories():
    if os.path.exists("./SplitDataset/"):
        return False
    os.mkdir("./SplitDataset/")
    os.mkdir("./SplitDataset/Audio/")
    os.mkdir("./SplitDataset/Spectrograms/")
    os.mkdir('./SplitDataset/Audio/test/')
    os.mkdir('./SplitDataset/Audio/train/') 
    os.mkdir('./SplitDataset/Spectrograms/test/')
    os.mkdir('./SplitDataset/Spectrograms/train/')

    
    for genre in GENRES:
        os.mkdir('./SplitDataset/Audio/test/'+genre)
        os.mkdir('./SplitDataset/Audio/train/'+genre)
        os.mkdir('./SplitDataset/Spectrograms/test/'+genre)
        os.mkdir('./SplitDataset/Spectrograms/train/'+genre)
    print("New Dataset Directory Created")
    return True
    
def SplitTestTrain():
    #Iterate over each genre in the dataset file
    for genre in GENRES:
        # Iterate over each file inside of the genre
        genre_folder    = os.path.join(FOLDER,genre)
        song_list       = os.listdir(genre_folder)
        np.random.shuffle(song_list)
        training_set    = song_list[0: int(.8* len(song_list))]
        test_set        = song_list[int(.8* len(song_list)):]
        for song in training_set:
            shutil.copy(genre_folder+'/'+song, f'./SplitDataset/Audio/train/{genre}')
        for song in test_set:
            shutil.copy(genre_folder+'/'+song, f'./SplitDataset/Audio/test/{genre}')
    print("Split created")

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled
    
# Split all the files in small excerpts
def SplitAudioFiles():
    # traing and test iteration
    for stage in ['train', 'test']:
            
         # dictionary to store mapping, labels, and MFCCs
        data = {
            "mapping": [],
            "labels": [],
            "mfcc": []
        }

        samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
        NUM_MFCC_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)
         
        json_path = stage+".json"
        dataset_path = "SplitDataset/Audio/"+stage

        # loop through all genre sub-folder
        for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):

            # ensure we're processing a genre sub-folder level
            if dirpath is not dataset_path:

                # save genre label (i.e., sub-folder name) in the mapping
                semantic_label = dirpath.split("/")[-1]
                data["mapping"].append(semantic_label)
                print("\nProcessing: {}".format(semantic_label))

                # process all audio files in genre sub-dir
                for f in filenames:

                    # load audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                
                
                    # process all segments of audio file
                    for d in range(NUM_SEGMENTS):

                        # calculate start and finish sample for current segment
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # extract mfcc
                        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
                        mfcc = mfcc.T

                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == NUM_MFCC_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print("{}, segment:{}".format(file_path, d+1))

        # save MFCCs to json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)


if __name__ == '__main__':
    np.random.seed(0)
    if CreateNewRepositories():
        SplitTestTrain()
    SplitAudioFiles()
