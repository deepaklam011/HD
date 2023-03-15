import torch
from transformers import BertTokenizer, BertModel



class MyCollate:
	def __init__(self, pad_idx):
		self.pad_idx = pad_idx
		# self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.tokenizer = BertTokenizer.from_pretrained('../bert-mlm')

	def __call__(self, batch):
		'''
		photo_ids: list of list [batch_size, num_images]
		
		returns:
			reviews: [reviews, sentences, words]
		'''
		# print(f'...inside MyCollate')
		sentences = [item[0] for item in batch]
		labels = [item[1] for item in batch]

		sentences = self._encode_for_bert(sentences)
		labels = torch.tensor(labels)

		return sentences, labels



	def _encode_for_bert(self, questions):
		
		questions = self.tokenizer.batch_encode_plus(questions, 
												return_tensors="pt", 
												padding=True,
												max_length=100,
												truncation=True
												)
		return questions



	def _pad_reviews(self, 
					 reviews, 
					 review_lengths, 
					 sentence_lengths
					 ):
		'''
		args:
			reviews:           [reviews, sentences, words]
			review_lengths:    [reviews]
			sentences_lengths: [reviews * sentences]
		
		returns:
			final: [reviews, max_sentences, max_words]
		'''
		batch_size = len(reviews)
		max_review_length = max(review_lengths)
		max_sentence_length = max(sentence_lengths)
		
		final = torch.zeros(batch_size, max_review_length, max_sentence_length, dtype=torch.int64)
		
		review_lengths = torch.zeros(batch_size, dtype=torch.int64)
		sentence_lengths = torch.zeros((batch_size, max_review_length), dtype=torch.int64)

		for i, review in enumerate(reviews):
			review_lengths[i] = len(review)
			for j, sentence in enumerate(review):
				final[i, j, :len(sentence)] = torch.Tensor(sentence)
				sentence_lengths[i, j] = len(sentence)
				
		return final, review_lengths, sentence_lengths
		
	def _pad_answers(self, answers, answers_lengths):
		batch_size = len(answers)
		max_answer_length = max(answers_lengths)
		
		final = torch.zeros(batch_size, max_answer_length, dtype=torch.int64)

		for i, answer in enumerate(answers):
			final[i, :len(answer)] = torch.Tensor(answer)
			# caption_lengths[i, j] = len(sentence)
		
		return final
		
	def _get_lengths(self, answers):
		answers_lengths = []
		# review_lengths = []
		# sentence_lengths = []
		for answer in answers:
			answers_lengths.append(len(answer))
			# for sentence in review:
				# sentence_lengths.append(len(sentence))

		answers_lengths = torch.tensor(answers_lengths, dtype=torch.int64)
		# sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.int64)

		return answers_lengths

	def _get_caption_lengths(self, captions_batch):
		caption_lengths = []
		for captions in captions_batch:
			for caption in captions:
				caption_lengths.append(len(caption))
		
		caption_lengths = torch.tensor(caption_lengths, dtype=torch.int64)
		
		return caption_lengths


