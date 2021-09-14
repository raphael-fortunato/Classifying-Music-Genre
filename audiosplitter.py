import os
import shutil
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Get the genres from the dataset
folder_name = "./dataset/gtzan/genres_original"
genres      = [name for name in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, name))]

# Make a new directory where we store the new files
def CreateNewRepositories():
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
    
def SplitTestTrain():
    #Iterate over each genre in the dataset file
    for genre in genres:

        # Iterate over each file inside of the genre
        genre_folder    = os.path.join(folder_name,f"{genre}")
        song_list       = os.listdir(genre_folder)
     
        np.random.shuffle(song_list)
        training_set    = song_list[0:80]
        test_set        = song_list[80:]
        for song in training_set:
            shutil.copy(genre_folder+'/'+song, f'./SplitDataset/Audio/train/{genre}')
        for song in test_set:
            shutil.copy(genre_folder+'/'+song, f'./SplitDataset/Audio/test/{genre}')
    
# Split all the files in small excerpts
def SplitAudioFiles():

    # traing and test iteration
    for stage in ['train', 'test']:
            
        working_folder = f'./SplitDataset/Audio/{stage}/'
        #Iterate over each genre in the dataset file
        for genre in genres:

            # Iterate over each file inside of the genre
            genre_folder    = os.path.join(working_folder,f"{genre}")
            song_list       = os.listdir(genre_folder)
            for filename in song_list:            

                # Get the song
                song_name   = os.path.join(genre_folder,f'{filename}')
                song        = AudioSegment.from_wav(song_name)

                # Clip 3 seconds excerpts
                for x in range(10):

                    # Clip information
                    clip_length     = (int(song.duration_seconds) / 10)*1000
                    clip            = song[x*clip_length: (x+1)*clip_length]
                    songName        = "_".join(song_name.split('/')[-1].split('.')[0:2]) + "_"
                    song_path       = f'./SplitDataset/Audio/{stage}/{genre}/{songName}{x}.wav'

                    # Export our newly made clip
                    clip.export(song_path, format='wav')

                    print("Saved: ", songName, x, end=" ")
                    # Load it in again to make a specotogram
                    y,sr = librosa.load(song_path,duration=3)
                    mels = librosa.feature.melspectrogram(y=y,sr=sr)
                    fig = plt.Figure()
                    canvas = FigureCanvas(fig)
                    p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
                    plt.savefig(f'./SplitDataset/Spectrograms/{stage}/{genre}/{songName}{x}.png')
                    

        print(" ")
        print("Done with genre: ", genre)


CreateNewRepositories()
SplitTestTrain()
SplitAudioFiles()
