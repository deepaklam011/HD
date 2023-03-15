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

from transformers import BertTokenizer, BertModel

def validate_batch(model, data, loss_fn, config, vocab=None):
	sentences, labels = data
	sentences = sentences.to(config.DEVICE)
	labels = labels.to(config.DEVICE)

	loss = 0
	prediction = []


	encodings, softmax = model(sentences=sentences)
	# print(f'encodings: {encodings.size()}')

	batch_correct, batch_samples = get_accuracy(encodings, labels)
	# batch_loss = loss_fn(encodings, labels)
	
	return encodings, softmax, labels, batch_correct, batch_samples


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
	output_json = []
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	y_true = np.array([])
	predictions = np.array([])

	val_correct, val_samples = 0, 0
	loop = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
	for batch_idx, data in loop:
		if(batch_idx == 4):
			break
		# continue
		log_soft, preds, y, batch_correct, batch_samples = validate_batch(
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

		sentences, labels = data
		input_ids = sentences['input_ids']
		input_id_list = input_ids[0].tolist()
		tokens = tokenizer.convert_ids_to_tokens(input_id_list)


		pretok_sent = ""
		for tok in tokens:
			if tok.startswith("##"):
				pretok_sent += tok[2:]
			else:
				pretok_sent += " " + tok
		pretok_sent = pretok_sent[1:]
		# print(pretok_sent)

		print(f'preds: {preds.shape}')
		# print(f'y: {y.shape}')
		preds = np.argmax(preds, axis = 1)
		print(f'preds: {preds[0]}')
		print(f'y: {y[0]}')
		print(pretok_sent)
		output_json.append({'pred': int(preds[0]), 'true': int(y[0]), 'sent': pretok_sent})
		# print(f'sentence: {tokens}')
		# print(f'preds: {preds.shape}')
		# print(f'y: {y.shape}')
		y_true = np.append(y_true, y)
		predictions = np.append(predictions, preds)
		# print(f'preds: {preds}')
		# print(f'y: {y}')
		loop.set_description(f"Inferencing..")
		loop.set_postfix(test_accuracy=val_acc.item())

	# print(f'>> precision; ')
	save_output_json_file(model, output_json_dict=output_json, dir='../saved_model_src', f=None, config=config)
	# print(f'>> {config.TEST_FILE}')
	# print(f'>> test_accuracy: {val_acc.item()*100}')
	# f1_score(y_true, predictions)
	# return np.mean(valid_losses)

if __name__ == "__main__":

	config = Config()

	print(f'>> {config.TEST_FILE}')
	valid_set, valid_loader = get_loader(config.PATH_TO_DATASET, 
										config.TEST_FILE, 
										1, 
										mode='test', 
										config=config
										)

	model = Model(config).to(config.DEVICE)

	load_model(model=model, 
				# decoder_gru=decoder_gru, 
				dir='../saved_model_src', 
				model_filename='model.pth', 
				# decoder_gru_filename='decoder_gru.pth'
				)

	# write_to_file = config.TEST_FILE.split('.')[0] + output + '.txt'

	loss_fn = nn.NLLLoss(reduction='mean')

	validate(
			valid_loader=valid_loader, 
			model=model, 
			# decoder_gru, 
			loss_fn=loss_fn,  
			# vocab, 
			config=config,
			)
