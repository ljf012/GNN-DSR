import math
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

day = 7

workdir = 'datasets/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Ciao', help='dataset name: Ciao/Epinions')
parser.add_argument('--test_prop', default=0.1, help='the proportion of data used for test')
args = parser.parse_args()

def update_id(*dataframes, colnames, mapping=None):
    """
    Map the values in the columns `colnames` of `dataframes` according to `mapping`.
    If `mapping` is `None`, a dictionary that maps the values in column `colnames[0]`
    of `dataframes[0]` to unique integers will be used.
    Note that values not appear in `mapping` will be mapped to `NaN`.

    Args
    ----
    dataframes : list[DataFrame]
        A list of dataframes.
    colnames: str, list[str]
        The names of columns.
    mapping: function, dict, optional
        Mapping correspondence.

    Returns
    -------
    DataFrame, list[DataFrame]
        A dataframe (if there is only one input dataframe) or a list of dataframes
        with columns in `colnames` updated according to `mapping`.
    """
    if type(colnames) is str:
        colnames = [colnames]
    if mapping is None:
        uniques = dataframes[0][colnames[0]].unique()
        mapping = {oid: i for i, oid in enumerate(uniques)}
    results = []
    for df in dataframes:
        columns = {}
        for name in colnames:
            if name in df.columns:
                columns[name] = df[name].map(mapping)
        df = df.assign(**columns)
        results.append(df)
    if len(results) == 1:
        return results[0]
    else:
        return results


# load data
if args.dataset == 'Ciao':
	click_f = np.loadtxt(workdir+'Ciao/rating_with_timestamp.txt', dtype = np.int32)
	min_timestamp = click_f[:,5].min()
	max_timestamp = click_f[:,5].max()
	print('min & max', min_timestamp, max_timestamp)
	time_id = [int(math.floor((t-min_timestamp) / (86400*day))) for t in click_f[:,5]]
	click_f[:,5] = time_id

	trust_f = np.loadtxt(workdir+'Ciao/trust.txt', dtype = np.int32)	

elif args.dataset == 'Epinions':
	click_f = np.loadtxt(workdir+'Epinions/rating_with_timestamp.txt', dtype = np.int32)
	min_timestamp = click_f[:,5].min()
	max_timestamp = click_f[:,5].max()
	print('min & max', min_timestamp, max_timestamp)
	time_id = [int(math.floor((t-min_timestamp) / (86400*day))) for t in click_f[:,5]]
	click_f[:,5] = time_id

	trust_f = np.loadtxt(workdir+'Epinions/trust.txt', dtype = np.int32)

elif args.dataset == 'delicious':
	df_clicks = pd.read_csv(
        workdir+'delicious/user_taggedbookmarks-timestamps.dat',
        sep='\t',
        skiprows=1,
        header=None,
        names=['userId', 'sessionId', 'itemId', 'timestamp'],
    )
	df_clicks['timestamp'] = pd.to_datetime(df_clicks.timestamp, unit='ms')
	session_len = df_clicks.groupby('sessionId', sort=False).size()
	long_sessions = session_len[session_len >= 2].index
	df_long = df_clicks[df_clicks.sessionId.isin(long_sessions)]

	df_edges = pd.read_csv(
        workdir+'delicious/user_contacts-timestamps.dat',
        sep='\t',
        skiprows=1,
        header=None,
        usecols=[0, 1],
        names=['follower', 'followee'],
    )
	count_df = df_long.nunique()
	print(count_df)

	df_long = update_id(df_long, colnames='userId')
	df_edges = update_id(df_edges, colnames='follower')
	df_edges = update_id(df_edges, colnames='followee')
	click_f = df_long.values
	trust_f = df_edges.values
else:
	pass 

click_list = []
trust_list = []

u_items_list = []
u_users_list = []
u_users_items_list = []
i_users_list = []

pos_u_items_list = []
pos_i_users_list = []

user_count = 0
item_count = 0
rate_count = 0
time_count = 0

for s in click_f:
	uid = s[0]
	iid = s[1]
	if args.dataset == 'Ciao':
		rating = s[3]
		tid = s[5]
	elif args.dataset == 'Epinions':
		rating = s[3]
		tid = s[5]
	elif args.dataset == 'delicious':
		iid = s[2]
		rating = 1
		tid = s[1]

	if uid > user_count:
		user_count = uid
	if iid > item_count:
		item_count = iid
	if rating > rate_count:
		rate_count = rating
	if tid > time_count:
		time_count = tid

	click_list.append([uid, iid, rating, tid])
	# print([uid, iid, rating, tid])	

pos_list = []
for i in range(len(click_list)):
	pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2], click_list[i][3]))


# remove duplicate items in pos_list because there are some cases where a user may have different rate scores on the same item.
pos_list = list(set(pos_list))

# filter user less than 5 items
pos_df = pd.DataFrame(pos_list, columns = ['uid', 'iid', 'rating', 'tid'])
filter_pos_list = []
user_in_set, user_out_set = set(), set()
for u in tqdm(range(user_count + 1)):
	hist = pos_df[pos_df['uid'] == u]
	if len(hist) < 5:
		user_out_set.add(u)
		continue
	user_in_set.add(u)
	u_items = hist['iid'].tolist()
	u_ratings = hist['rating'].tolist()
	u_timestamp = hist['tid'].tolist()
	filter_pos_list.extend([(u, iid, rating, tid) for iid, rating, tid in zip(u_items, u_ratings, u_timestamp)])
print('user in and out size: ', len(user_in_set), len(user_out_set))
print('data size before and after filtering: ', len(pos_list), len(filter_pos_list))


# train, valid and test data split
print('test prop: ', args.test_prop)
pos_list = filter_pos_list

random.shuffle(pos_list)
num_test = int(len(pos_list) * args.test_prop)
test_set = pos_list[:num_test]
valid_set = pos_list[num_test:2 * num_test]
train_set = pos_list[2 * num_test:]


print('Train samples: {}, Valid samples: {}, Test samples: {}, Total samples: {}'.format(len(train_set), len(valid_set), len(test_set), len(pos_list)))

with open(workdir + args.dataset + '/dataset_filter5.pkl', 'wb') as f:
	pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)


pos_df = pd.DataFrame(pos_list, columns = ['uid', 'iid', 'rating', 'tid'])
train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'rating', 'tid'])
valid_df = pd.DataFrame(valid_set, columns = ['uid', 'iid', 'rating', 'tid'])
test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'rating', 'tid'])

click_df = pd.DataFrame(click_list, columns = ['uid', 'iid', 'rating', 'tid'])
train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')
pos_df = pos_df.sort_values(axis = 0, ascending = True, by = 'uid')

"""
u_items_list: 
	Store the iid and corresponding rating of the item 
	each user has interacted with, or [(0,0,0)] if none
"""
for u in tqdm(range(user_count + 1)):
	hist = train_df[train_df['uid'] == u]
	u_items = hist['iid'].tolist()
	u_ratings = hist['rating'].tolist()
	u_timestamp = hist['tid'].tolist()
	if u_items == []:
		u_items_list.append([(0,0,0)])
	else:
		u_items_list.append([(iid, rating, tid) for iid, rating, tid in zip(u_items, u_ratings, u_timestamp)])


train_df = train_df.sort_values(axis = 0, ascending = True, by = 'iid')

"""
i_users_list: 
	Store the users associated with each item 
	and their ratings, or [(0,0,0)] if none
example:
[
	iid:(uid,rating,tid)
	00:[(11,3,70),(222,4,71),(33,5,75)],
    01:[(21,5,12),(234,4,12)],
    02:[(35,4,12)],
    03:[(5445,5,12),(633,1,12)],
    04:[(61,5,12),(10,2,12),(91,4,12),(11,3,12)],
]
"""
userful_item_set = set()
for i in tqdm(range(item_count + 1)):
	hist = train_df[train_df['iid'] == i]
	i_users = hist['uid'].tolist()
	i_ratings = hist['rating'].tolist()
	u_timestamp = hist['tid'].tolist()
	if i_users == []:
		i_users_list.append([(0,0,0)])
	else:
		i_users_list.append([(uid, rating, tid) for uid, rating, tid in zip(i_users, i_ratings, u_timestamp)])
		userful_item_set.add(i)

print('item size before and after filtering: ', item_count, len(userful_item_set))

with open(workdir + args.dataset + '/effective_users_and items_filter5.pkl', 'wb') as f:
	pickle.dump(list(user_in_set), f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(list(userful_item_set), f, pickle.HIGHEST_PROTOCOL)


'''
correlative item
i_items_list：Store the item iid of the same type for each item.
i_items_users_list：Store a list of user uid's for each item of the same type.
'''
top_N = 100
i_items_list = []
i_items_users_list = []

i_ratings_list = []
for id in i_users_list:
    i_ratings_list.append([rate[1] for rate in id])


for i in tqdm(range(item_count + 1)):
	ij_list = []
	
	# Calculating cosine similarity
	for j in range(0, item_count + 1, random.randint(1, 1000)):
		if i != j:
			
			rating_len = min(len(i_ratings_list[i]), len(i_ratings_list[j]))
			# if rating_len > 5:
			x = torch.Tensor(i_ratings_list[i][:rating_len]).cuda()
			y = torch.Tensor(i_ratings_list[j][:rating_len]).cuda()
			cosine = torch.cosine_similarity(x, y, dim=0)
			ij_list.append([j, cosine])

	ij_df = pd.DataFrame(ij_list, columns=['iid', 'cos'])
	ij_df = ij_df.sort_values(by='cos', ascending = False)
	ij_df = ij_df.head(top_N)

	i_items = ij_df['iid'].unique().tolist()
	if i_items == []:
		i_items_list.append([0])
		i_items_users_list.append([[(0,0,0)]])
	else:
		i_items_list.append(i_items)
		ii_users = []
		for iid in i_items:
			ii_users.append(i_users_list[iid])
		i_items_users_list.append(ii_users)


count_f_origin, count_f_filter = 0,0
for s in trust_f:
	uid = s[0]
	fid = s[1]
	count_f_origin += 1
	if uid > user_count or fid > user_count:
		continue
	if uid in user_out_set or fid in user_out_set:
		continue
	trust_list.append([uid, fid])
	count_f_filter += 1

print('u-u relation filter size changes: ', count_f_origin, count_f_filter)
trust_df = pd.DataFrame(trust_list, columns = ['uid', 'fid'])
trust_df = trust_df.sort_values(axis = 0, ascending = True, by = 'uid')


"""
u_users_list: Store the user uid of each user that has interacted with it.
u_users_items_list: Store a list of iid's for each of the user's friends
"""
count_0, count_1 = 0,0
for u in tqdm(range(user_count + 1)):
	hist = trust_df[trust_df['uid'] == u]
	u_users = hist['fid'].unique().tolist()
	if u_users == []:
		u_users_list.append([0])
		u_users_items_list.append([[(0,0,0)]])
		count_0 += 1
	else:
		u_users_list.append(u_users)
		uu_items = []
		for uid in u_users:
			uu_items.append(u_items_list[uid])
		u_users_items_list.append(uu_items)
		count_1 += 1
print('trust user with items size: ', count_0, count_1)


with open(workdir + args.dataset + '/list_filter5.pkl', 'wb') as f:
	pickle.dump(u_items_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(u_users_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(u_users_items_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_users_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_items_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_items_users_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump((user_count, item_count, rate_count, time_count), f, pickle.HIGHEST_PROTOCOL)
