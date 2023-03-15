import torch
import torch.nn as nn

import torch.nn.functional as F

from bert_encoder import BERTEncoder
from classifier import Classifier

from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class Model(nn.Module):
    def __init__(self, 
                vocab=None, 
                config=None):
        super(Model, self).__init__()

        print(f'Loading model from {__name__}.py')

        self.name = __name__
        self.config = config

        # self.sentence_encoder_bert = BERTEncoder(config=config)
        # self.classifier = Classifier(config=config)
        # self.tokenizer_sbert = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
        self.sbert = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')
        # encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')


        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)


    def forward(self, 
                setup=setup,
                punchline=punchline,
                non_punchline=non_punchline,
                non_relation=non_relation,
                relation=relation
        ):

        with torch.no_grad():
            setup_output = model(**setup)
            punchline_output = model(**punchline)
            non_punchline_output = model(**non_punchline)

        setup_embedding = mean_pooling(setup_output, setup['attention_mask'])
        setup_embedding = F.normalize(setup_embedding, p=2, dim=1)

        punchline_embedding = mean_pooling(punchline_output, punchline['attention_mask'])
        punchline_embedding = F.normalize(punchline_embedding, p=2, dim=1)

        non_punchline_embedding = mean_pooling(non_punchline_output, setup['attention_mask'])
        non_punchline_embedding = F.normalize(non_punchline_embedding, p=2, dim=1)


        return output

