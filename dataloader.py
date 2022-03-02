from torch.utils.data import Dataset

class DSRDataset(Dataset):
	def __init__(self, data, u_items_list, u_users_list, u_users_items_list, i_users_list, i_items_list, i_items_users_list):
		self.data = data
		self.u_items_list = u_items_list
		self.u_users_list = u_users_list
		self.u_users_items_list = u_users_items_list
		self.i_users_list = i_users_list
		self.i_items_list = i_items_list
		self.i_items_users_list = i_items_users_list

	def __getitem__(self, index):
		uid = self.data[index][0]
		iid = self.data[index][1]
		rating = self.data[index][2]
		tid = self.data[index][3]
		u_items = self.u_items_list[uid]
		u_users = self.u_users_list[uid]
		u_users_items = self.u_users_items_list[uid]
		i_users = self.i_users_list[iid]
		i_items = self.i_items_list[iid]
		i_items_users = self.i_items_users_list[iid]

		return (uid, iid, rating, tid), u_items, u_users, u_users_items, i_users, i_items, i_items_users

	def __len__(self):
		return len(self.data)
