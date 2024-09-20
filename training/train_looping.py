import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.my_tools import paint_smi_matrixs

torch.manual_seed(1)  # random seed. We not yet optimization it.

class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, path='best.pt', trace_func=print, saveckpt=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            path (str): Path for saving the checkpoint model. 
                        Default: 'checkpoint.pt'
            trace_func (function): Trace print function.
                                   Default: print
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.trace_func = trace_func
        self.saveckpt = saveckpt

    def __call__(self, val_loss, model, epoch, train_losses, valid_losses, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.saveckpt:
                self.save_checkpoint(val_loss, model, epoch, train_losses, valid_losses, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.saveckpt:
                self.save_checkpoint(val_loss, model, epoch, train_losses, valid_losses, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, train_losses, valid_losses, optimizer):
        '''Saves model when validation loss decreases.'''

        # TODO: Implement the model saving.
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'trainLosses': train_losses,
                    'valLosses': valid_losses
                }
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss

def train_loop(n_epochs, model, train_set, valid_set, train=True, valid=True, inference=False, batch_size=1, lr=1e-6,
               ckpt_name='ckpt', lastckpt=None, saveckpt=False, log_dir='scalar', device_ids=[0], mae_error=False, use_wandb=False, fold_index=0, patience=10):
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=f'checkpoint/{ckpt_name}_best.pt', saveckpt=saveckpt)

    # Initialize wandb if use_wandb is True
    if use_wandb:
        import wandb
        wandb.init(project="repetitive-action-counting", entity="cares", name="TransRAC_CV" + str(fold_index))
    
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    currEpoch = 0
    trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=16)
    validloader = DataLoader(valid_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=16)
    model = nn.DataParallel(model.to(device), device_ids=device_ids)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    milestones = [i for i in range(0, n_epochs, 40)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.8)  # three step decay

    writer = SummaryWriter(log_dir=os.path.join('log/', log_dir))
    scaler = GradScaler()

    if lastckpt is not None:
        print("loading checkpoint")
        checkpoint = torch.load(lastckpt)
        # currEpoch = checkpoint['epoch']
        currEpoch = 0
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    lossMSE = nn.MSELoss()
    lossSL1 = nn.SmoothL1Loss()

    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        trainLosses = []
        validLosses = []
        trainLoss1 = []
        validLoss1 = []
        trainOBO = []
        validOBO = []
        trainMAE = []
        validMAE = []
        predCount = []
        Count = []

        if train:
            pbar = tqdm(trainloader, total=len(trainloader))
            batch_idx = 0
            for input, target in pbar:
                with autocast():
                    model.train()
                    optimizer.zero_grad()
                    acc = 0
                    input = input.type(torch.FloatTensor).to(device)
                    density = target.type(torch.FloatTensor).to(device)
                    count = torch.sum(target, dim=1).type(torch.FloatTensor).round().to(device)
                    output, matrixs = model(input)
                    predict_count = torch.sum(output, dim=1).type(torch.FloatTensor).to(device)
                    predict_density = output
                    loss1 = lossMSE(predict_density, density)
                    loss2 = lossSL1(predict_count, count)
                    loss3 = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                            predict_count.flatten().shape[0]  # mae
                    loss = loss1
                    if mae_error:
                        loss += loss3

                    # calculate MAE or OBO
                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    OBO = acc / predict_count.flatten().shape[0]
                    trainOBO.append(OBO)
                    MAE = loss3.item()
                    trainMAE.append(MAE)

                    train_loss = loss.item()
                    train_loss1 = loss1.item()
                    trainLosses.append(train_loss)
                    trainLoss1.append(train_loss1)
                    batch_idx += 1
                    pbar.set_postfix({'Epoch': epoch,
                                      'loss_train': train_loss,
                                      'Train MAE': MAE,
                                      'Train OBO ': OBO})

                    if batch_idx % 10 == 0:
                        writer.add_scalars('train/loss', {"loss": np.mean(trainLosses)}, epoch * len(trainloader) + batch_idx)
                        writer.add_scalars('train/MAE', {"MAE": np.mean(trainMAE)}, epoch * len(trainloader) + batch_idx)
                        writer.add_scalars('train/OBO', {"OBO": np.mean(trainOBO)}, epoch * len(trainloader) + batch_idx)
                        
                        # Log to wandb if use_wandb is True
                        if use_wandb:
                            wandb.log({"train/loss": np.mean(trainLosses),
                                       "train/MAE": np.mean(trainMAE),
                                       "train/OBO": np.mean(trainOBO)})

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        if valid and epoch % 5 == 0:
            with torch.no_grad():
                batch_idx = 0
                pbar = tqdm(validloader, total=len(validloader))
                for input, target in pbar:
                    model.eval()
                    acc = 0
                    input = input.type(torch.FloatTensor).to(device)
                    density = target.type(torch.FloatTensor).to(device)
                    count = torch.sum(target, dim=1).type(torch.FloatTensor).round().to(device)

                    output, sim_matrix = model(input)
                    predict_count = torch.sum(output, dim=1).type(torch.FloatTensor).to(device)
                    predict_density = output

                    loss1 = lossMSE(predict_density, density)
                    loss2 = lossSL1(predict_count, count)
                    loss3 = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                            predict_count.flatten().shape[0]  # mae
                    loss = loss1
                    if mae_error:
                        loss += loss3
                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    OBO = acc / predict_count.flatten().shape[0]
                    validOBO.append(OBO)
                    MAE = loss3.item()
                    validMAE.append(MAE)
                    train_loss = loss.item()
                    train_loss1 = loss1.item()
                    validLosses.append(train_loss)
                    validLoss1.append(train_loss1)

                     # Apply early stopping at the end of each validation step
                    valid_loss_mean = np.mean(validLosses)
                    early_stopping(valid_loss_mean, model, epoch, trainLosses, validLosses, optimizer)

                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        break  # Break out of the loop if early stopping is triggered


                    batch_idx += 1
                    pbar.set_postfix({'Epoch': epoch,
                                      'loss_valid': train_loss,
                                      'Valid MAE': MAE,
                                      'Valid OBO ': OBO})

                writer.add_scalars('valid/loss', {"loss": np.mean(validLosses)}, epoch)
                writer.add_scalars('valid/OBO', {"OBO": np.mean(validOBO)}, epoch)
                writer.add_scalars('valid/MAE', {"MAE": np.mean(validMAE)}, epoch)
                
                # Log to wandb if use_wandb is True
                if use_wandb:
                    wandb.log({"valid/loss": np.mean(validLosses),
                               "valid/OBO": np.mean(validOBO),
                               "valid/MAE": np.mean(validMAE)})

        scheduler.step()
    
        writer.add_scalars('learning rate', {"learning rate": optimizer.state_dict()['param_groups'][0]['lr']}, epoch)
        writer.add_scalars('epoch_trainMAE', {"epoch_trainMAE": np.mean(trainMAE)}, epoch)
        writer.add_scalars('epoch_trainOBO', {"epoch_trainOBO": np.mean(trainOBO)}, epoch)
        writer.add_scalars('epoch_trainloss', {"epoch_trainloss": np.mean(trainLosses)}, epoch)
        
        # Log to wandb if use_wandb is True
        if use_wandb:
            wandb.log({"learning rate": optimizer.state_dict()['param_groups'][0]['lr'],
                       "epoch_trainMAE": np.mean(trainMAE),
                       "epoch_trainOBO": np.mean(trainOBO),
                       "epoch_trainloss": np.mean(trainLosses)})



        writer.add_scalars('learning rate', {"learning rate": optimizer.state_dict()['param_groups'][0]['lr']}, epoch)
        writer.add_scalars('epoch_trainMAE', {"epoch_trainMAE": np.mean(trainMAE)}, epoch)
        writer.add_scalars('epoch_trainOBO', {"epoch_trainOBO": np.mean(trainOBO)}, epoch)
        writer.add_scalars('epoch_trainloss', {"epoch_trainloss": np.mean(trainLosses)}, epoch)

    # Finish the wandb run if use_wandb is True
    if use_wandb:
        wandb.finish()