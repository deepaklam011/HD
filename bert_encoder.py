import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

class BERTEncoder(nn.Module):
	def __init__(self, config):
		super(BERTEncoder, self).__init__()
		self.config = config
		self.name = __name__

		# self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.bert = BertModel.from_pretrained('../bert-mlm')


	def forward(self, input):

		outputs = self.bert(**input, return_dict=True)

		last_hidden_states = outputs.last_hidden_state

		sentence_representations = last_hidden_states[:, 0, :]

		return sentence_representations

