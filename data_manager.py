import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from hparams import hparams
import pdb

'''
Load Dataset (divided into train/validate/test sets)
* audio data : saved as segments in npy file
* labels : 50-d labels in csv file
'''

class AudioDataset(Dataset):
    def __init__(self, x, y_mid, y_emo):
        self.x = x
        self.y_mid = y_mid
        self.y_emo = y_emo

    def __getitem__(self, index):
        return self.x[index], self.y_mid[index], self.y_emo[index]

    def __len__(self):
        return len(self.x)


def load_dataset(mode, hparams):
    x = []
    y_mid = []
    y_emo = []
    print("dataset mode: ",mode)
    if mode == 'train':
        annotation_file = Path(hparams.annotations)/'train5000.csv'
    elif mode == 'valid':
        annotation_file = Path(hparams.annotations)/'valid5000.csv'
    elif mode == 'test':
        annotation_file = Path(hparams.annotations)/'valid5000.csv'
    elif mode == 'train360':
        annotation_file = Path(hparams.annotations)/'train300.csv'
    elif mode == 'valid360':
        annotation_file = Path(hparams.annotations)/'valid300.csv'


    annotations_frame = pd.read_csv(annotation_file) # df

    for index in range(len(annotations_frame)):

        # audio_path = str(annotations_frame.iloc[index]['song_id'])+'.npy'
        audio_path = str(annotations_frame.iloc[index]['Number'])+'.npy'

        audio = np.load(hparams.feature_path + '/' + audio_path)
        label_mid = np.array(annotations_frame.iloc[index][hparams.mid_feature], dtype=np.float64)
        label_emo = np.array(annotations_frame.iloc[index][hparams.emotion_label], dtype=np.float64)

        label_mid = torch.from_numpy(label_mid)
        label_emo = torch.from_numpy(label_emo)

        x.append(audio)
        y_mid.append(label_mid)
        y_emo.append(label_emo)
    return x,y_mid,y_emo


def get_dataloader(hparams):
    x_train, y_train_mid, y_train_emo = load_dataset('train360', hparams)
    x_valid, y_valid_mid, y_valid_emo = load_dataset('valid360', hparams)
    x_test, y_test_mid, y_valid_emo = load_dataset('valid360', hparams)

    train_set = AudioDataset(x_train, y_train_mid, y_train_emo)
    vaild_set = AudioDataset(x_valid, y_valid_mid, y_valid_emo)
    test_set = AudioDataset(x_test, y_test_mid, y_valid_emo)

    train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(vaild_set, batch_size=hparams.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=hparams.batch_size, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader

