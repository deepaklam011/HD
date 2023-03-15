import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import numpy as np
# ! pip install spacy
import spacy  # for tokenizer
import json

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from PIL import Image  # Load img
import torchvision.transforms as transforms

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.nn.utils.rnn import pack_padded_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
import importlib
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(seed)

from config import Config
# from vocab import Vocabulary
from evaluate import validate
# from dataset import VistaNetDataset
# from collate_fn import MyCollate
# from sentence_encoder_gru import SentenceEncoderGRU
# from sentence_encoder_cnn import SentenceEncoderCNN
# from caption_encoder import CaptionEncoder
# from review_encoder import ReviewEncoder
# from final_classifier import FinalClassifier

from utils import get_loader, save_model, save_checkpoint, load_checkpoint
# from decoder_gru import DecoderGRU


from model import Model
# from model_image import Model
# from model_image_gcn import Model
# from model_image_gat import Model
# from model_image_gat_co_att import Model

def train_batch(model, data, model_optimizer, loss_fn, config, vocab=None):
	sentences, labels = data
	sentences = sentences.to(config.DEVICE)
	labels = labels.to(config.DEVICE)

	loss = 0
	# print(f'questions: {questions.size()}')
	# print(f'answers: {answers.size()}')
	
	model_optimizer.zero_grad()

	encodings, softmax_output = model(sentences=sentences) 


	# print(f'>> encodings: {encodings.size()}')

	loss = loss_fn(encodings, labels)

	loss.backward()

	if config.CLIP_GRADIENTS:
	  _ = nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_AT)
	  # _ = nn.utils.clip_grad_norm_(decoder_gru.parameters(), config.CLIP_AT)

	model_optimizer.step()

	return loss.item()


def train(train_loader, 
			valid_loader, 
			model=None, 
			decoder_gru=None, 
			model_optimizer=None, 
			decoder_gru_optimizer=None, 
			loss_fn=None, 
			train_vocab=None, 
			f=None, 
			config=None, 
			my_lr_scheduler=None):

	patience, limit = config.PATIENCE, 0
	best_val_acc = 0
	best_val_loss = np.inf
	best_bleu_score = -1
	val_bleu_score = 0
	all_losses = []
	for epoch in range(config.START_EPOCH, config.EPOCHS):
		model.train()
		# decoder_gru.train()
		epoch_loss = []
		epoch_acc = []
		epoch_samples = 0
		epoch_correct = 0
		# val_epoch_samples = 0
		# val_epoch_correct = 0
		# print(len(train_loader))
		loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
		for batch_idx, data in loop:
			# print(data[0])
			# print(data[1])
			# print(data[2])
			# if(batch_idx == 1):
			# 	break
			# batch_loss, batch_correct, batch_samples = train_batch(model, data, decoder_gru, model_optimizer, decoder_gru_optimizer, loss_fn, config, train_vocab)
			batch_loss = train_batch(model=model, 
									# decoder_gru, 
									data=data, 
									model_optimizer=model_optimizer, 
									# decoder_gru_optimizer, 
									loss_fn=loss_fn, 
									config=config
									# vocab=train_vocab
									)
			# if(batch_idx == 0):
			# 	break
			# epoch_correct += batch_correct
			# epoch_samples += batch_samples
			# batch_acc = batch_correct/batch_samples
			epoch_loss.append(batch_loss)
			# epoch_acc.append(batch_acc.item())
			# epoch_accuracy = epoch_correct/epoch_samples

			loop.set_description(f"Epoch [{epoch+1}/{config.EPOCHS}]")
			loop.set_postfix(epoch_loss=np.mean(epoch_loss), running_loss=np.mean(epoch_loss[-200:]))
		# break

		if epoch % 1 == 0:
			val_loss = validate(valid_loader=valid_loader, 
								model=model, 
								# decoder_gru=decoder_gru, 
								loss_fn=loss_fn, 
								# train_vocab=train_vocab,
								config=config
								)
			if best_val_loss >= val_loss:
				best_val_loss = val_loss
				print('Validation loss improved...')
				save_model(model, dir='../checkpoints-bert-pretrained', config=config)
				# save_checkpoint(
				# 				model=model, 
				# 				# decoder_gru=decoder_gru, 
				# 				epoch=epoch, 
				# 				model_optimizer=model_optimizer, 
				# 				# decoder_gru_optimizer=decoder_gru_optimizer,
				# 				scheduler=my_lr_scheduler,
				# 				config=config
				# 				)
				limit = 0
			else:
				limit += 1
				print(f'Validation loss did not improve in last {limit} epochs.\n')

			if limit == patience:
				print(f'Stopping early')
				return


	# 	avg_test_acc = test(model, train_vocab, config=config, f=f)
	# 	print_some_stats(f, 
	# 					 flag='imp', 
	# 					 epoch=epoch, 
	# 					 epoch_acc=epoch_correct/epoch_samples, 
	# 					 epoch_loss=np.mean(epoch_loss),
	# 					 val_loss=val_loss,
	# 					 val_acc=val_acc, 
	# 					 avg_test_acc=avg_test_acc)
	# 	config.SAVE_COUNTER += 1
	# 	save_model(model, config=config)


	# 	if best_val_loss > val_loss:
	# 		best_val_loss = val_loss
	# 		print('Validation loss improved...')
	# 		save_model(model, decoder_gru, config=config)
	# # 		limit = 0
	# 	else:
	# 		limit += 1
	# 		print(f'Validation loss did not improve in last {limit} epochs.\n')

		# if best_bleu_score < val_bleu_score:
		# 	best_bleu_score = val_bleu_score
		# 	print('Validation bleu score improved...')
		# 	save_model(model, decoder_gru, config=config)
		# 	limit = 0
		# else:
		# 	limit += 1
		# 	print(f'Validation bleu score did not improve in last {limit} epochs.\n')

		# if limit == patience:
		# 	print(f'Stopping early')
		# 	return
	# 	my_lr_scheduler.step()
	# 	all_losses.append(np.mean(epoch_loss))

	# return all_losses


def main():

	config = Config()

	train_set, train_loader = get_loader(config.PATH_TO_DATASET, 
										config.TRAIN_FILE, 
										config.BATCH_SIZE, 
										mode='train', 
										# vocab=vocab,
										config=config
										)

	valid_set, valid_loader = get_loader(config.PATH_TO_DATASET, 
										config.VALIDATION_FILE, 
										config.BATCH_SIZE_EVALUATE, 
										mode='val', 
										# vocab=vocab,
										config=config
										)

	model = Model(config).to(config.DEVICE)
	# decoder_gru = DecoderGRU(vocab, model.embedding_layer, config).to(config.DEVICE)

	# model_optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
	model_optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

	# model_optimizer = optim.Adam([{'params': model.base.parameters()},
	# 							{'params': model.gat.parameters(), 'lr': config.LEARNING_RATE * 10}],
	# 							lr=config.LEARNING_RATE)



	# decoder_gru_optimizer = optim.Adam(decoder_gru.parameters(), lr=config.LEARNING_RATE * config.DECODER_LEARNING_RATIO)
	model_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=model_optimizer, gamma=config.EXP_LR_DECAY)
	# loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

	# if config.CONTINUE_TRAINING:
	# 	load_checkpoint(model=model, 
	# 					# decoder_gru=decoder_gru, 
	# 					epoch=config.START_EPOCH,
	# 					model_optimizer=model_optimizer, 
	# 					# decoder_gru_optimizer=decoder_gru_optimizer,
	# 					scheduler=model_lr_scheduler)

	# 	print(f'>>>> {config.START_EPOCH}')

	loss_fn = nn.NLLLoss(reduction='mean')

	# with open(output_file, 'w+') as f:
	# train(train_loader, valid_loader, model, model_optimizer, loss_fn, vocab, f, config, my_lr_scheduler)
	train(train_loader=train_loader, 
			valid_loader=valid_loader, 
			model=model, 
			# decoder_gru=decoder_gru, 
			model_optimizer=model_optimizer, 
			# decoder_gru_optimizer=decoder_gru_optimizer, 
			loss_fn=loss_fn, 
			# train_vocab=vocab, 
			# vocab, 
			config=config, 
			my_lr_scheduler=model_lr_scheduler)
	# f.close()
	# train(train_loader, valid_loader, config=config)

main()





