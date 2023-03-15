import os
from torch.utils.data import DataLoader, Dataset
import json
import torch
import numpy as np
import pickle
import pandas as pd

# from vocab import Vocabulary
# from collate_fn import MyCollate

class HumorDataset(Dataset):
	def __init__(self, 
				root_dir, 
				data_file, 
				transform=None,
				freq_threshold=3, 
				num_images=None, 
				mode=None, 
				vocab=None,
				config=None
				):

		self.root_dir = root_dir
		self.path_to_file = os.path.join(root_dir, data_file)
		self.data = self._read_data(self.path_to_file)
		self.mode = mode

		self.setup, self.punchline, self.non_punchline =\
					self._get_stuff(self.data)


	def __len__(self):
		return len(self.data)


	def __getitem__(self, index):

		setup = self.setup[index]
		punchline = self.punchline[index]
		non_punchline = self.non_punchline[index]

		return setup, punchline, non_punchline


	def _read_data(self, file):
		data = pd.read_csv(file)
		return data


	def _get_stuff(self, data):

		setup = 'setup'
		punchline = 'punchline'
		non_punchline = 'non_punchline'

		setup = data[setup].to_list()
		punchline = data[punchline].to_list()
		non_punchline = data[non_punchline].to_list()

		return setup, punchline, non_punchline

