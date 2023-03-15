import numpy as np
import torch
from tqdm import tqdm
from utils import *
import torch.nn as nn

from config import Config
# from vocab import Vocabulary

from torch.utils.data import DataLoader, Dataset
import importlib
from tqdm import tqdm

from utils import get_loader
# from decoder_gru import DecoderGRU
from utils import calculate_bleu_score

from model import Model
# from model_image import Model
# from model_image_gcn import Model
# from model_image_gat import Model
# from model_image_gat_co_att import Model


def validate_batch(model, data, loss_fn, config, vocab=None):
	sentences, labels = data
	sentences = sentences.to(config.DEVICE)
	labels = labels.to(config.DEVICE)

	loss = 0
	prediction = []


	encodings, softmax_output = model(sentences=sentences) 

	batch_correct, batch_samples = get_accuracy(encodings, labels)
	batch_loss = loss_fn(encodings, labels)
	
	return batch_correct, batch_samples, batch_loss.item()


def validate(valid_loader, 
			model=None, 
			# decoder_gru=None, 
			loss_fn=None, 
			train_vocab=None, 
			f=None, 
			config=None, 
		):
	model.eval()
	# decoder_gru.eval()
	all_losses = []
	# bleu_scores = []
	# model.train()
	valid_losses = []

	val_correct, val_samples = 0, 0
	loop = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
	for batch_idx, data in loop:
		if(batch_idx == 100):
			break
		# continue
		batch_correct, batch_samples, batch_loss = validate_batch(model=model, 
																data=data, 
																loss_fn=loss_fn, 
																config=config, 
																)
		val_correct += batch_correct
		val_samples += batch_samples
		val_acc = val_correct/val_samples
		valid_losses.append(batch_loss)
		# bleu_scores.append(bleu_score)

		loop.set_description(f"Validating")
		loop.set_postfix(valid_acc=val_acc.item(), valid_loss=np.mean(valid_losses))

	return np.mean(valid_losses)

if __name__ == "__main__":

	config = Config()


	valid_set, valid_loader = get_loader(config.PATH_TO_DATASET, 
										config.VALIDATION_FILE, 
										config.BATCH_SIZE_EVALUATE, 
										mode='val', 
										# vocab=vocab,
										config=config
										)

	model = Model(config).to(config.DEVICE)

	loss_fn = nn.NLLLoss(reduction='mean')

	validate(
			valid_loader=valid_loader, 
			model=model, 
			# decoder_gru, 
			loss_fn=loss_fn,  
			# vocab, 
			config=config,
		)
