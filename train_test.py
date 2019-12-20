'''
train_test.py
A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_manager
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils

from sklearn.metrics import r2_score

import numpy as np

from model import *
from hparams import hparams


class Runner(object):
    def __init__(self, hparams):
        self.model = net(hparams)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=hparams.factor, patience=hparams.patience, verbose=True)
        self.learning_rate = hparams.learning_rate
        self.stopping_rate = hparams.stopping_rate
        self.device = hparams.device
        self.criterion = torch.nn.MSELoss()


    # Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
    def run(self, dataloader, mode='train'):
        self.model.train() if mode is 'train' else self.model.eval()
        epoch_loss = 0
        all_prediction_mid = []
        all_label_mid = []
        all_prediction_emo = []
        all_label_emo = []

        Tensor = torch.FloatTensor
        model, Tensor = utils.set_device(self.model, Tensor, self.device)

        for batch, (x, y_mid, y_emo) in enumerate(dataloader):
            audio = x.type(Tensor)
            label_mid = y_mid.type(Tensor)
            label_emo = y_emo.type(Tensor)

            mid_y, emo_y = model(audio)
            loss_mid = self.criterion(mid_y, label_mid)
            loss_emo = self.criterion(emo_y, label_emo)

            all_prediction_mid.extend(mid_y.cpu().detach().numpy())
            all_label_mid.extend(label_mid.cpu().detach().numpy())

            all_prediction_emo.extend(emo_y.cpu().detach().numpy())
            all_label_emo.extend(label_emo.cpu().detach().numpy())

            loss = loss_mid + loss_emo

            if mode is 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss/len(dataloader.dataset)

        return all_prediction_mid, all_label_mid, all_prediction_emo, all_label_emo, avg_loss

    # Early stopping function for given validation loss
    def early_stop(self, loss, epoch):
        self.scheduler.step(loss, epoch)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        stop = self.learning_rate < self.stopping_rate
        return stop

def main():    
    train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
    runner = Runner(hparams)

    for epoch in range(hparams.num_epochs):
        
        train_y_pred_mid, train_y_true_mid,train_y_pred_emo, train_y_true_emo, train_loss=runner.run(train_loader, 'train')
        valid_y_pred_mid, valid_y_true_mid,valid_y_pred_emo, valid_y_true_emo, val_loss= runner.run(valid_loader, 'eval')

        train_Coeff_mid = r2_score(train_y_true_mid, train_y_pred_mid)
        val_Coeff_mid = r2_score(valid_y_true_mid, valid_y_pred_mid)

        train_Coeff_emo = r2_score(train_y_pred_emo, train_y_true_emo)
        val_Coeff_emo = r2_score(valid_y_pred_emo, valid_y_true_emo)
        print("[Epoch %d/%d] [Train Loss: %.4f] [Train Coeff: %.4f] [Valid Loss: %.4f] [Valid Coeff: %.4f]" %
              (epoch + 1, hparams.num_epochs, train_loss, train_Coeff_emo, val_loss, val_Coeff_emo))

        if runner.early_stop(val_loss, epoch + 1):
            break

    print("Training Finished")

    test_y_pred_mid, test_y_true_mid, test_y_pred_emo, test_y_true_emo, test_loss = runner.run(test_loader, 'eval')
    test_Coeff_mid = r2_score(test_y_true_mid, test_y_pred_mid)
    test_Coeff_emo = r2_score(test_y_pred_emo, test_y_true_emo)
    print ('Test Coeff emo: {:.4f} \n'. format(test_Coeff_emo))

if __name__ == '__main__':
    main()