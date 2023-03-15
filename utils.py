from dataset import HumorDataset
from torch.utils.data import DataLoader, Dataset
from collate_fn import MyCollate
import torch
import numpy as np
import os
from nltk.translate.bleu_score import sentence_bleu
import json
from sklearn.metrics import precision_recall_fscore_support, classification_report

def calculate_bleu_score(prediction, answer, vocab, print_it=True):
	# prediction: list
	# answer: str
	# print(prediction)
	# print(answer)

	answer = [answer[0].split()]
	# answer[0].append("<EOS>")
	score_1 = sentence_bleu(answer, prediction, weights=(1, 0, 0, 0))
	score_2 = sentence_bleu(answer, prediction, weights=(0, 1, 0, 0))
	score_3 = sentence_bleu(answer, prediction, weights=(0, 0, 1, 0))
	if print_it:
		# print(score)
		print(prediction)
		print(answer)
		print(score_1)
	return score_1, score_2, score_3


def save_output_json_file(model, output_json_dict, dir='../checkpoints', f=None, config=None):
	
	# sub_dir = f'{model.name}'
	sub_dir = config.TRAIN_FILE.split('.')[0]
	if not os.path.exists(os.path.join(dir, sub_dir)):
		os.makedirs(os.path.join(dir, sub_dir))

	filename = config.TEST_FILE.split('.')[0] + '.json'

	print(output_json_dict)
	with open(os.path.join(dir, sub_dir, filename), 'w') as outfile:
		json.dump(output_json_dict, outfile)

	print('Saved file as {}'.format(filename))



def save_model(model, decoder_gru=None, dir='../checkpoints-classification', dir2='../configs', f=None, config=None):
	# if (lr == None) or (wd == None):
		# filename = os.path.join(dir, f'{model.name}.pth')
	# else:
		# filename = os.path.join(dir, f'{model.name}_lr_{lr}_wd_{wd}.pth')

	# sub_dir = f'{model.name}'
	sub_dir = config.TRAIN_FILE.split('.')[0]

	if not os.path.exists(os.path.join(dir, sub_dir)):
		os.makedirs(os.path.join(dir, sub_dir))

	# save model
	model_name = f'{model.name}'
	# filename = os.path.join(dir, f'{model_name}.pth')
	filename = os.path.join(dir, sub_dir, f'{model_name}.pth')
	torch.save(model.state_dict(), filename)
	print('Saved model as {}'.format(filename))
	
	# save model.bert
	model_name = f'{model.sentence_encoder_bert.name}'
	# filename = os.path.join(dir, f'{model_name}.pth')
	filename = os.path.join(dir, sub_dir, f'{model_name}.pth')
	model.sentence_encoder_bert.bert.save_pretrained(os.path.join(dir, sub_dir))
	print('Saved model as {}'.format(filename))

	# save decoder_gru
	# model_name = f'{decoder_gru.name}'
	# # filename = os.path.join(dir, f'{model_name}.pth')
	# filename = os.path.join(dir, sub_dir, f'{model_name}.pth')
	# torch.save(decoder_gru.state_dict(), filename)
	# print('Saved model as {}'.format(filename))

	# if config.SAVE_CONFIG is True:
	#     with open(os.path.join(dir2, f'{model_name}.txt'), 'w+') as f1:
	#         # f1.write(f'config: {vars(config)}')
	#         pretty_print_configs(f1, config)
	#     f1.close()
		# filename = os.path.join(dir, f'{model_name}.pth')






def load_model(model, 
				# decoder_gru, 
				dir, 
				model_filename='model.pth', 
				# decoder_gru_filename='decoder_gru.pth'
				):

	sub_dir = f'{model.name}'
	path = os.path.join(dir, sub_dir, model_filename)
	print(f'>> {path}')
	model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
	model.eval()
	print(f'Loaded {model_filename} successfully...')
	
	# path = os.path.join(dir, sub_dir, decoder_gru_filename)
	# print(f'>> {path}')
	# decoder_gru.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
	# decoder_gru.eval()
	# print(f'Loaded {decoder_gru_filename} successfully...')
	
	model.sentence_encoder_bert.bert.from_pretrained(os.path.join(dir, sub_dir))
	print(f'Loaded {model.sentence_encoder_bert.name} successfully...')


def get_accuracy(probs, ratings, output_file=None):
	indices = torch.max(probs, dim=1)
	correct = (indices[1] == ratings).float().sum()  
	acc = correct / probs.size(0)
	samples = probs.size(0)

	return correct, samples


def f1_score(y_true, y_pred):
	# precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred, average='micro')
	# print(f'>> precision: {precision}')
	# print(f'>> recall: {recall}')
	# print(f'>> f1_score: {f1_score}')
	# print(f'>> support: {support}')
	y_true = y_true.astype(int)
	y_pred = y_pred.astype(int)
	# print(y_true, y_pred)
	print(classification_report(y_true, y_pred))


def get_loader(root_dir, 
				json_file,
				batch_size,
				mode,
				vocab=None,
				config=None
				):

	dataset = HumorDataset(root_dir, json_file, mode=mode, config=config)
	# pad_idx = dataset.vocab.stoi["<PAD>"]
	if mode == 'train':
		shuffle = True
	else:
		shuffle = True
	loader = DataLoader(dataset=dataset, 
						batch_size=batch_size, 
						shuffle=shuffle,
						collate_fn=MyCollate(pad_idx=0),
						num_workers=2,
						drop_last=True
						)
	return dataset, loader



	# model_optimizer = optim.Adam([{'params': model.gat.parameters(), 'lr': config.LEARNING_RATE * 10}, 
	#                                 {'params': model.parameters(), 'lr': self.LEARNING_RATE,}])


def load_checkpoint(model, 
					decoder_gru, 
					epoch,
					model_optimizer, 
					decoder_gru_optimizer,
					scheduler, 
					dir='../checkpoints', 
					model_filename='model.pth', 
					decoder_gru_filename='decoder_gru.pth'
					):

	sub_dir = f'{model.name}'
	model_name = 'checkpoint_1.pth'
	path = os.path.join(dir, sub_dir, model_name)

	print(f'Loaded {model_name} successfully...')

	checkpoint = torch.load(path)
	epoch = checkpoint['epoch']
	model.load_state_dict(checkpoint['model_state_dict'])
	decoder_gru.load_state_dict(checkpoint['decoder_gru_state_dict'])
	model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
	decoder_gru_optimizer.load_state_dict(checkpoint['decoder_gru_optimizer_state_dict'])
	scheduler.load_state_dict(checkpoint['scheduler'])
	model.train()
	decoder_gru.train()
	# print(f'>> {path}')
	# model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
	# model.eval()
	
	
	# path = os.path.join(dir, sub_dir, decoder_gru_filename)
	# print(f'>> {path}')
	# decoder_gru.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
	# decoder_gru.eval()
	# print(f'Loaded {decoder_gru_filename} successfully...')
	
	# model.sentence_encoder_bert.bert.from_pretrained(os.path.join(dir, sub_dir, ))
	# print(f'Loaded {model.sentence_encoder_bert.name} successfully...')




def save_checkpoint(
					model, 
					decoder_gru, 
					epoch, 
					model_optimizer, 
					decoder_gru_optimizer,
					scheduler, 
					dir='../checkpoints', 
					dir2='../configs', 
					f=None, 
					config=None
					):

	sub_dir = f'{model.name}'

	if not os.path.exists(os.path.join(dir, sub_dir)):
		os.makedirs(os.path.join(dir, sub_dir))

	checkpoint_dict = { 
						'epoch': epoch + 1, 
						'model_state_dict': model.state_dict(), 
						'decoder_gru_state_dict': decoder_gru.state_dict(), 
						'model_optimizer_state_dict': model_optimizer.state_dict(), 
						'decoder_gru_optimizer_state_dict': decoder_gru_optimizer.state_dict(), 
						'scheduler': scheduler
						}

	path = os.path.join(dir, sub_dir, f'checkpoint_{epoch+1}.pth')
	torch.save(checkpoint_dict, path)
	print(f'Saved model as checkpoint_{epoch+1}.pth')

