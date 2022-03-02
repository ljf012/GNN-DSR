import os
import time
import json
import argparse
import pickle
import random
import numpy as np
from torch._C import device
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import collate_fn
from model import GNN_DSR
from dataloader import DSRDataset


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Ciao', help='dataset: Ciao/Epinions')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=128, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=20, help='the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1234, help='the number of random seed to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--device', type=int, default=0, help='the index of GPU device (-1 for CPU)')
parser.add_argument('--test', action='store_true', help='test model')
args = parser.parse_args()
print(args)
print(torch.cuda.is_available)

device = (torch.device('cpu') if args.device < 0 else torch.device(f'cuda:{args.device}'))

torch.backends.cudnn.benchmark=True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


here = os.path.dirname(os.path.abspath(__file__))
fn = 'weights/' + args.dataset


if not os.path.exists(fn):
    os.mkdir(fn)


def main():
    print('Loading data...')
    with open('datasets/' + args.dataset + '/dataset_filter5.pkl', 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)


    with open('datasets/' + args.dataset + '/list_filter5.pkl', 'rb') as f:
        u_items_list = pickle.load(f)
        u_users_list = pickle.load(f)
        u_users_items_list = pickle.load(f)
        i_users_list = pickle.load(f)
        i_items_list = pickle.load(f)
        i_items_users_list = pickle.load(f)
        (user_count, item_count, rate_count, time_count) = pickle.load(f)

    
    train_data = DSRDataset(train_set, u_items_list,  u_users_list, u_users_items_list, i_users_list, i_items_list, i_items_users_list)
    valid_data = DSRDataset(valid_set, u_items_list,  u_users_list, u_users_items_list, i_users_list, i_items_list, i_items_users_list)
    test_data = DSRDataset(test_set, u_items_list, u_users_list,  u_users_items_list, i_users_list, i_items_list, i_items_users_list)
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn, drop_last=True)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn, drop_last=True)

    model = GNN_DSR(user_count+1, item_count+1, rate_count+1, args.embed_dim, args.batch_size, args.device).to(device)

    if args.test:
        print('Load checkpoint and testing...')
        ckpt = torch.load('%s/random_best_checkpoint.pth.tar' %fn, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        mae, rmse = validate(test_loader, model)
        print("Test: MAE: {:.4f}, RMSE: {:.4f}".format(mae, rmse))
        return

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()


    valid_loss_list, test_loss_list = [],[]

    for epoch in tqdm(range(args.epoch)):

        # train for one epoch
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 100)

        mae, rmse = validate(valid_loader, model)
        valid_loss_list.append([mae, rmse])

        test_mae, test_rmse = validate(test_loader, model)
        test_loss_list.append([test_mae, test_rmse])

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if epoch == 0:
            best_rmse = rmse
            best_mae = mae
        elif rmse < best_rmse or mae < best_mae:
            best_rmse = rmse
            best_mae = mae
            torch.save(ckpt_dict, '%s/random_best_checkpoint.pth.tar' %fn)

        print('Epoch {} validation: MAE: {:.4f}, RMSE: {:.4f}, Best MAE: {:.4f}, test_MAE: {:.4f}, test_RMSE: {:.4f}'.format(epoch, mae, rmse, best_mae, test_mae, test_rmse))


        with open('%s/random_valid_loss_list.txt' %fn, 'w') as f:
            f.write(json.dumps(valid_loss_list))

        with open('%s/random_test_loss_list.txt' %fn, 'w') as f:
            f.write(json.dumps(test_loss_list))


def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (uids, iids, ratings, tids, u_items, u_users, u_users_items, i_users, i_items, i_items_users) in tqdm(enumerate(train_loader), total=len(train_loader)):
        uids = uids.to(device)
        iids = iids.to(device)
        ratings = ratings.to(device)
        tids = tids.to(device)
        u_items = u_items.to(device)
        u_users = u_users.to(device)
        u_users_items = u_users_items.to(device)

        i_users = i_users.to(device)
        i_items = i_items.to(device)
        i_items_users = i_items_users.to(device)

        optimizer.zero_grad()
        outputs = model(uids, iids, u_items, u_users, u_users_items, i_users, i_items, i_items_users)

        loss = criterion(outputs, ratings.unsqueeze(1))
        loss.backward()
        optimizer.step() 

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            mean_loss = sum_epoch_loss / (i + 1)
            print('[TRAIN WWW] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, mean_loss,
                  len(uids) / (time.time() - start)))
        
        start = time.time()


def validate(valid_loader, model):
    model.eval()
    errors = []
    with torch.no_grad():
        for uids, iids, ratings, tids, u_items, u_users, u_users_items, i_users, i_items, i_items_users in tqdm(valid_loader):
            uids = uids.to(device)
            iids = iids.to(device)
            ratings = ratings.to(device)
            tids = tids.to(device)
            u_items = u_items.to(device)
            u_users = u_users.to(device)
            u_users_items = u_users_items.to(device)

            i_users = i_users.to(device)
            i_items = i_items.to(device)
            i_items_users = i_items_users.to(device)

            preds = model(uids, iids, u_items, u_users, u_users_items, i_users, i_items, i_items_users)
            error = torch.abs(preds.squeeze(1) - ratings)
            errors.extend(error.data.cpu().numpy().tolist())
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.power(errors, 2)))
    return mae, rmse


if __name__ == '__main__':
    main()
