from src.checkpoint import Checkpoint

from src.loss import ELBO

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR

from math import exp
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decay(x):
    return 0.01 + (0.99)*(0.9999)**x

class Trainer:
    def __init__(self, 
                 learning_rate=1e-3,
                 KL_rate=0.9999,
                 free_bits=256,
                 sampling_rate=2000,
                 batch_size=512, 
                 print_every=1000, 
                 checkpoint_every=10000,
                 checkpoint_dir='checkpoint',
                 output_dir='outputs'):
        self.learning_rate = learning_rate
        self.KL_rate = KL_rate
        self.free_bits = free_bits
        self.optimizer=None
        self.scheduler=None
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.print_every = print_every
        self.checkpoint_every = checkpoint_every
        self.output_dir = output_dir
        
    def inverse_sigmoid(self,step):
        """
        Compute teacher forcing probability with inverse sigmoid
        """
        k = self.sampling_rate
        if k == None:
            return 0
        if k == 1.0:
            return 1
        return k/(k + exp(step/k))
        
    def KL_annealing(self, step, start, end):
        return end + (start - end)*(self.KL_rate)**step
    
    def compute_loss(self, step, model, batch, use_teacher_forcing=True):
        batch.to(device)
        pred, mu, sigma, z = model(batch, use_teacher_forcing)
        elbo, kl = ELBO(pred, batch, mu, sigma, self.free_bits)
        kl_weight = self.KL_annealing(step, 0, 0.2)
        return kl_weight*elbo, kl
        
    def train_batch(self, iter, model, batch):
        self.optimizer.zero_grad()
        use_teacher_forcing = self.inverse_sigmoid(iter)
        elbo, kl = self.compute_loss(iter, model, batch, use_teacher_forcing)
        self.scheduler.step()
        elbo.backward()
        self.optimizer.step()
        return elbo.item(), kl.item()
        
    def train_epochs(self, model, start_epoch, iter, end_epoch, train_data, val_data=None):
        train_loss, val_loss = [], []
        train_kl,  val_kl = [], []
        for epoch in range(start_epoch, end_epoch):
            batch_loss, batch_kl = [], []
            model.train()
            
            for idx, batch in enumerate(train_data):
                print(batch_loss)
                batch = batch.transpose(0, 1).squeeze()
                batch.to(device)
                elbo, kl = self.train_batch(iter, model, batch)
                batch_loss.append(elbo)
                batch_kl.append(kl)
                iter += 1
                
                if iter%self.print_every == 0:
                    loss_avg = torch.mean(torch.tensor(batch_loss))
                    div = torch.mean(torch.tensor(batch_kl))
                    print('Epoch: %d, iteration: %d, Average loss: %.4f, KL Divergence: %.4f' % (epoch, iter, loss_avg, div))
                
                if iter%self.checkpoint_every == 0:
                    self.save_checkpoint(model, epoch, iter)
            
            train_loss.append(torch.mean(torch.tensor(batch_loss)))
            train_kl.append(torch.mean(torch.tensor(batch_kl)))
            
            self.save_checkpoint(model, epoch, iter)
            
            if val_data is not None:
                batch_loss, batch_kl = [], []
                with torch.no_grad():
                    model.eval()
                    for idx, batch in enumerate(val_data):
                        batch.to(device)
                        batch = batch.transpose(0, 1).squeeze()
                        elbo, kl = self.compute_loss(iter, model, batch, False)
                        batch_loss.append(elbo)
                        batch_kl.append(kl)
                    val_loss.append(torch.mean(torch.tensor(batch_loss)))
                    val_kl.append(torch.mean(torch.tensor(batch_kl)))
                loss_avg = torch.mean(torch.tensor(val_loss))
                div = torch.mean(torch.tensor(val_kl))
                print('Validation')
                print('Epoch: %d, iteration: %d, Average loss: %.4f, KL Divergence: %.4f' % (epoch, iter, loss_avg, div))
                    
        torch.save(open('outputs/train_loss_musicvae_batch', 'wb+'), torch.tensor(train_loss))
        torch.save(open('outputs/val_loss_musicvae_batch', 'wb+'), torch.tensor(val_loss))
        torch.save(open('outputs/train_kl_musicvae_batch', 'wb+'), torch.tensor(train_kl))
        torch.save(open('outputs/val_kl_musicvae_batch', 'wb+'), torch.tensor(val_kl))
        
    def save_checkpoint(self, model, epoch, iter):
        print('Saving checkpoint')
        Checkpoint(model=model,
                    epoch=epoch,
                    step=iter,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    samp_rate=self.sampling_rate,
                    KL_rate=self.KL_rate,
                    free_bits=self.free_bits).save(self.output_dir)
        print('Checkpoint Successful')
        
    def load_checkpoint(self):
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.output_dir)
        resume_checkpoint  = Checkpoint.load(latest_checkpoint_path)
        model              = resume_checkpoint.model
        epoch              = resume_checkpoint.epoch
        iter               = resume_checkpoint.step
        self.scheduler     = resume_checkpoint.scheduler
        self.optimizer     = resume_checkpoint.optimizer
        self.sampling_rate = resume_checkpoint.samp_rate
        self.KL_rate       = resume_checkpoint.KL_rate
        self.free_bits     = resume_checkpoint.free_bits
        return model, epoch, iter
    
    def train(self, model, train_data, optimizer, epochs, resume=False, val_data=None):
        if resume:
            model, epoch, iter = self.load_checkpoint()
        else:
            if optimizer is None:
                self.optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
                self.scheduler = LambdaLR(self.optimizer, decay)
                
            epoch = 1
            iter = 0
            print(model)
            print(self.optimizer)
            print(self.scheduler)
            print('Starting epoch %d' % epoch)

        model.to(device)
        self.train_epochs(model, epoch, iter, epoch+epochs, train_data, val_data)
                
                        
                        
            
            
            
            
                
                
