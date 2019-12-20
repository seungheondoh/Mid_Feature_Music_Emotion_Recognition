''' Run this file to process raw audio '''
import os, errno
import numpy as np
import torch 
import librosa
from pathlib import Path
from hparams import hparams
from scipy.fftpack import fft
import pandas as pd
from sklearn.model_selection import train_test_split

def dataset_divide(hparams):
    df_5000_meta = pd.read_csv(os.path.join(hparams.meta5000,'metadata.csv'), sep=';')
    df_5000_meta.columns = ['song_id','Source-genre','ID at file source','Title','Artist','Album_name']
    df_5000 = pd.read_csv(os.path.join(hparams.meta5000,'annotations.csv'))
    df_5000.columns = ['song_id', 'melody', 'articulation', 'rhythm_complexity','rhythm_stability', 'dissonance', 'atonality', 'mode']
    df_5000_final = pd.merge(df_5000, df_5000_meta, on='song_id')
    df_stratify = df_5000_final['Source-genre']
    df5000_X_train, df5000_test, _, _ = train_test_split(df_5000_final, df_stratify, test_size=0.08, random_state=0, stratify=df_stratify)

    df_360_label = df_5000_final[df_5000_final['Source-genre'] == 'soundtracks']
    df_emo = pd.read_excel(hparams.label_path_360)
    df_mid = df_360_label[['ID at file source','melody','articulation','rhythm_complexity','rhythm_stability','dissonance','atonality','mode']] 
    df_mid = df_mid.rename(columns = {'ID at file source':'Number'})
    df_mid = df_mid.astype({'Number': 'int64'})

    Soundtracks = pd.merge(df_emo,df_mid, on='Number')
    target = Soundtracks['TARGET']
    df360_X_train, df360_test, _, _ = train_test_split(Soundtracks, target, test_size=0.2, random_state=0, stratify=target)

    print("df5000_X_train col= %s \t  df360_X_train= %s  "% (df5000_X_train.columns, df360_X_train.columns))
    
    df5000_X_train.to_csv(os.path.join(hparams.annotations, r"train5000.csv"), mode='w')
    df5000_test.to_csv(os.path.join(hparams.annotations, r"valid5000.csv"), mode='w')
    df360_X_train.to_csv(os.path.join(hparams.annotations, r"train300.csv"), mode='w')
    df360_test.to_csv(os.path.join(hparams.annotations, r"valid300.csv"), mode='w')

def paper_spectrogram(file_name):
    y, sr = librosa.load(file_name, hparams.sample_rate, offset=float(np.random.choice(5, 1)) ,duration=10)
    frame = librosa.util.frame(y,frame_length=2048, hop_length=700)    
    all_frep = []
    for i in frame.T:
        freq = fft(i, 300)
        all_frep.append(freq[:149])
    S = np.array(all_frep)
    linear_S = np.log10(1+10*np.abs(S))
    return linear_S

def melspectrogram(file_name, hparams):
    y, sr = librosa.load(file_name, hparams.sample_rate, offset=float(np.random.choice(5, 1)) ,duration=10)
    S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)
    mel_basis = librosa.filters.mel(hparams.sample_rate, n_fft=hparams.fft_size, n_mels=hparams.num_mels)
    mel_S = np.dot(mel_basis, np.abs(S))
    mel_S = np.log10(1+10*mel_S)
    mel_S = mel_S.T
    return mel_S

def resize_array(array, length):
    resize_array = np.zeros((length, array.shape[1]))
    if array.shape[0] >= length:
        resize_array = array[:length]
    else:
        resize_array[:array.shape[0]] = array

    return resize_array

def save_audio_to_npy(dataset_path,feature_path):
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    audio_path = os.path.join(dataset_path)
    audios = [audio for audio in os.listdir(audio_path)]
    for audio in audios:
        audio_abs = os.path.join(audio_path,audio)
        try:
            feature = melspectrogram(audio_abs, hparams)
            if len(feature) < 100:
                print("Feature length is less than 100")
        except:
            print("Cannot load audio {}".format(audio))
            continue
        feature = resize_array(feature, hparams.feature_length)
        fn = audio.split(".")[0]
        print(Path(feature_path) / (fn + '.npy'))
        np.save(Path(feature_path) / (fn + '.npy'), feature)

if __name__ == '__main__':
    save_audio_to_npy(hparams.dataset_path, hparams.mel_feature_path)
    dataset_divide(hparams)
    print("Finish")

                    