import os
import time
import shutil
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import skimage.io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Get the genres from the dataset
folder_name = "./dataset/gtzan/genres_original"
genres      = [name for name in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, name))]

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

    
    for genre in genres:
        os.mkdir('./SplitDataset/Audio/test/'+genre)
        os.mkdir('./SplitDataset/Audio/train/'+genre)
        os.mkdir('./SplitDataset/Spectrograms/test/'+genre)
        os.mkdir('./SplitDataset/Spectrograms/train/'+genre)
    print("New Dataset Directory Created")
    return True
    
def SplitTestTrain():
    #Iterate over each genre in the dataset file
    for genre in genres:
        # Iterate over each file inside of the genre
        genre_folder    = os.path.join(folder_name,genre)
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
            
        working_folder = f'./SplitDataset/Audio/{stage}/'
        #Iterate over each genre in the dataset file
        for genre in genres:

            # Iterate over each file inside of the genre
            genre_folder    = os.path.join(working_folder,genre)
            song_list       = os.listdir(genre_folder)
            for filename in song_list:            

                # Get the song
                song_name   = os.path.join(genre_folder,filename)
                song        = AudioSegment.from_wav(song_name)

                # Clip 3 seconds excerpts
                for x in range(10):

                    # Clip information
                    clip_length     = (int(song.duration_seconds) / 10)*1000
                    clip            = song[x*clip_length: (x+1)*clip_length]
                    songName        = "_".join(song_name.split('/')[-1].split('.')[0:2]) + "_"
                    song_path       = f"./SplitDataset/Audio/{stage}/{genre}/{songName}{x}.wav"
                    if os.path.isfile(song_path):
                        print("file exists")
                        continue
                    # Export our newly made clip
                    clip.export(song_path, format='wav')

                    print("Saved: ", songName, x, end=" ")
                    print()
                    # Load it in again to make a specotogram
                    start = time.time()
                    y,sr = librosa.load(song_path,duration=3)
                    mels = librosa.feature.melspectrogram(y=y,sr=sr)
                    mels = np.log(mels + 1e-9) # add small number to avoid log(0)

                    # min-max scale to fit inside 8-bit range
                    img = scale_minmax(mels, 0, 255).astype(np.uint8)
                    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
                    img = 255-img # invert. make black==more energy

                    # save as PNG
                    skimage.io.imsave(f"./SplitDataset/Spectrograms/{stage}/{genre}/{songName}{x}.png",
                            img)
                    time_elapsed = time.time() - start
                    print('completed in {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
                        

        print(" ")
        print("Done with genre: ", genre)


if __name__ == '__main__':
    np.random.seed(0)
    if CreateNewRepositories():
        SplitTestTrain()
    SplitAudioFiles()
