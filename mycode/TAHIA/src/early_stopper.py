"""
Early stop provided by DGL
"""
import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=10, path='es_checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.path = path

    def step(self, acc, model, epoch, auc, trec, tpre, acc2):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.auc = auc
            self.acc2 = acc2
            self.trec = trec
            self.tpre = tpre
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            # print(
            #     f'EarlyStopping counter: {self.counter}/{self.patience}, best_val_f1:{self.best_score:.4f}, Auc:{self.auc:.4f}, Precision:{self.tpre:.4f}, Recall:{self.trec:.4f}, Accuracy:{self.acc2:.4f} at E{self.best_epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            
            self.auc = auc
            self.acc2 = acc2
            self.trec = trec
            self.tpre = tpre
            self.save_checkpoint(model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)
