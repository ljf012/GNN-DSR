import torch
import random
import numpy as np

truncate_len = 30
truncate_len_i = 30
soc_len = 30
cor_len = 30

"""
Ciao dataset info:
Avg number of items rated per user: 38.3
Avg number of users interacted per user: 2.7
Avg number of users connected per item: 16.4
"""

def collate_fn(batch_data):
    """This function will be used to pad the graph to max length in the batch
       It will be used in the Dataloader
    """
    uids, iids, ratings, tids = [], [], [], []
    u_items, u_users, u_users_items, i_users, i_items, i_items_users = [], [], [], [], [], []
    u_items_len, u_users_len, i_users_len, i_items_len = [], [], [], []

    for data, u_items_u, u_users_u, u_users_items_u, i_users_i, i_items_i, i_items_users_i in batch_data:

        (uid, iid, rating, tid) = data
        uids.append(uid)
        iids.append(iid)
        ratings.append(rating)
        tids.append(tid)

        # user-items    
        if len(u_items_u) <= truncate_len:
            temp = np.array(u_items_u)
            temp = temp[np.lexsort(temp.T)].tolist()
            u_items.append(temp)
        else:
            temp = np.array(random.sample(u_items_u, truncate_len))
            temp = temp[np.lexsort(temp.T)].tolist()
            u_items.append(temp)
        u_items_len.append(min(len(u_items_u), truncate_len))


        # user-users and user-users-items
        if len(u_users_u) < soc_len:
            tmp_users = [item for item in u_users_u]
            tmp_users.append(uid)
            u_users.append(tmp_users)
            u_u_items = [] 
            for uui in u_users_items_u:
                if len(uui) < truncate_len:
                    temp = np.array(uui)
                    temp = temp[np.lexsort(temp.T)].tolist()
                    u_u_items.append(temp)
                else:
                    temp = np.array(random.sample(uui, truncate_len))
                    temp = temp[np.lexsort(temp.T)].tolist()
                    u_u_items.append(temp)
            # self -loop
            u_u_items.append(u_items[-1])
            u_users_items.append(u_u_items)

        else:
            sample_index = random.sample(list(range(len(u_users_u))), soc_len-1)
            tmp_users = [u_users_u[si] for si in sample_index]
            tmp_users.append(uid)
            u_users.append(tmp_users)

            u_users_items_u_tr = [u_users_items_u[si] for si in sample_index]
            u_u_items = [] 
            for uui in u_users_items_u_tr:
                if len(uui) < truncate_len:
                    temp = np.array(uui)
                    temp = temp[np.lexsort(temp.T)].tolist()
                    u_u_items.append(temp)
                else:
                    temp = np.array(random.sample(uui, truncate_len))
                    temp = temp[np.lexsort(temp.T)].tolist()
                    u_u_items.append(temp)
            u_u_items.append(u_items[-1])
            u_users_items.append(u_u_items)

        u_users_len.append(min(len(u_users_u)+1, soc_len))


        # item-users
        if len(i_users_i) <= truncate_len_i:
            temp = np.array(i_users_i)
            temp = temp[np.lexsort(temp.T)].tolist()
            i_users.append(i_users_i)
        else:
            temp = np.array(random.sample(i_users_i, truncate_len_i))
            temp = temp[np.lexsort(temp.T)].tolist()
            i_users.append(temp)
        i_users_len.append(min(len(i_users_i), truncate_len_i))

        # item-items and item-items-users
        if len(i_items_i) < cor_len:
            tmp_items = [user for user in i_items_i]
            tmp_items.append(iid)
            i_items.append(tmp_items)
            i_i_users = []
            for iiu in i_items_users_i:
                if len(iiu) < truncate_len_i:
                    temp = np.array(iiu)
                    temp = temp[np.lexsort(temp.T)].tolist()
                    i_i_users.append(temp)
                else:
                    temp = np.array(random.sample(iiu, truncate_len_i))
                    temp = temp[np.lexsort(temp.T)].tolist()
                    i_i_users.append(temp)
            # self -loop
            i_i_users.append(i_users[-1])
            i_items_users.append(i_i_users)
        
        else:
            sample_index = random.sample(list(range(len(i_items_i))), cor_len-1)
            tmp_items = [i_items_i[si] for si in sample_index]
            tmp_items.append(iid)
            i_items.append(tmp_items)

            i_items_users_i_tr = [i_items_users_i[si] for si in sample_index]
            i_i_users = []
            for iiu in i_items_users_i_tr:
                if len(iiu) < truncate_len_i:
                    temp = np.array(iiu)
                    temp = temp[np.lexsort(temp.T)].tolist()
                    i_i_users.append(temp)
                else:
                    temp = np.array(random.sample(iiu, truncate_len_i))
                    temp = temp[np.lexsort(temp.T)].tolist()
                    i_i_users.append(temp)
            i_i_users.append(i_users[-1])
            i_items_users.append(i_i_users)

        i_items_len.append(min(len(i_items_i)+1, cor_len))


    batch_size = len(batch_data)

    # padding
    u_items_maxlen = max(u_items_len)
    u_users_maxlen = max(u_users_len)
    i_users_maxlen = max(i_users_len)
    i_items_maxlen = max(i_items_len)


    
    u_item_pad = torch.zeros([batch_size, u_items_maxlen, 3], dtype=torch.long)
    for i, ui in enumerate(u_items):
        u_item_pad[i, :len(ui), :] = torch.LongTensor(ui)
    

    u_user_pad = torch.zeros([batch_size, u_users_maxlen], dtype=torch.long)
    for i, uu in enumerate(u_users):
        u_user_pad[i, :len(uu)] = torch.LongTensor(uu)

    
    u_user_item_pad = torch.zeros([batch_size, u_users_maxlen, u_items_maxlen, 3], dtype=torch.long)
    for i, uu_items in enumerate(u_users_items):
        for j, ui in enumerate(uu_items):
            u_user_item_pad[i, j, :len(ui), :] = torch.LongTensor(ui)

   
    i_user_pad = torch.zeros([batch_size, i_users_maxlen, 3], dtype=torch.long)
    for i, iu in enumerate(i_users):
        i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)


    i_item_pad = torch.zeros([batch_size, i_items_maxlen], dtype=torch.long)
    for i, ii in enumerate(i_items):
        i_item_pad[i, :len(ii)] = torch.LongTensor(ii)

    
    i_item_user_pad = torch.zeros([batch_size, i_items_maxlen, i_users_maxlen, 3], dtype=torch.long)
    for i, ii_users in enumerate(i_items_users):
        for j, iu in enumerate(ii_users):
            i_item_user_pad[i, j, :len(iu), :] = torch.LongTensor(iu)


    return torch.LongTensor(uids), torch.LongTensor(iids), torch.FloatTensor(ratings), torch.IntTensor(tids), \
            u_item_pad, u_user_pad, u_user_item_pad, i_user_pad, i_item_pad, i_item_user_pad
