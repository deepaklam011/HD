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
from utils import calculate_bleu_score, get_accuracy, f1_score

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
	# print(f'encodings: {encodings.size()}')

	batch_correct, batch_samples = get_accuracy(encodings, labels)
	# batch_loss = loss_fn(encodings, labels)
	
	return encodings, labels, batch_correct, batch_samples


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
	# all_losses = []
	# bleu_scores = []
	# model.train()
	# valid_losses = []

	y_true = np.array([])
	predictions = np.array([])

	val_correct, val_samples = 0, 0
	loop = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
	for batch_idx, data in loop:
		if(batch_idx == 2):
			break
		# continue
		preds, y, batch_correct, batch_samples = validate_batch(
													model=model, 
													data=data, 
													loss_fn=loss_fn, 
													config=config, 
													)
		val_correct += batch_correct
		val_samples += batch_samples
		val_acc = val_correct/val_samples
		# valid_losses.append(batch_loss)
		# bleu_scores.append(bleu_score)
		# print(f'preds: {preds.size()}')
		# print(f'y: {y.size()}')

		preds = preds.detach().cpu().numpy()
		y = y.detach().cpu().numpy()
		# print(f'preds: {preds.shape}')
		# print(f'y: {y.shape}')
		preds = np.argmax(preds, axis = 1)
		# print(f'preds: {preds.shape}')
		# print(f'y: {y.shape}')
		y_true = np.append(y_true, y)
		predictions = np.append(predictions, preds)
		loop.set_description(f"Inferencing..")
		loop.set_postfix(test_accuracy=val_acc.item())

	# print(f'>> precision; ')

	print(f'>> {config.TEST_FILE}')
	print(f'>> test_accuracy: {val_acc.item()*100}')
	f1_score(y_true, predictions)
	# return np.mean(valid_losses)

if __name__ == "__main__":

	config = Config()

	print(f'>> {config.TEST_FILE}')
	valid_set, valid_loader = get_loader(config.PATH_TO_DATASET, 
										config.TEST_FILE, 
										config.BATCH_SIZE_EVALUATE, 
										mode='test', 
										config=config
										)

	model = Model(config).to(config.DEVICE)

	load_model(model=model, 
				# decoder_gru=decoder_gru, 
				dir='../checkpoints-bert-pretrained', 
				model_filename='model.pth', 
				# decoder_gru_filename='decoder_gru.pth'
				)


	loss_fn = nn.NLLLoss(reduction='mean')

	validate(
			valid_loader=valid_loader, 
			model=model, 
			# decoder_gru, 
			loss_fn=loss_fn,  
			# vocab, 
			config=config,
			)
