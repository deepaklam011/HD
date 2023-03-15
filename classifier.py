import torch
import torch.nn as nn
import torch.nn.functional as F

# from hyperparameters import *

class Classifier(nn.Module):
    def __init__(self, 
                 review_emb_dim=None,
                 layer_1_out=256,
                 num_classes=2,
                 config=None
                ):
        super(Classifier, self).__init__()
        self.config = config

        # when using GAT
        self.fc1 = nn.Linear(768, num_classes)

        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        # print(f'>> input: {input.size()}')
        output = input
        # output = self.activation(output)
        raw_unnormalized_scores = self.fc1(output)
        softmax_output = self.softmax(raw_unnormalized_scores)
        raw_unnormalized_scores = self.log_softmax(raw_unnormalized_scores)

        return raw_unnormalized_scores, softmax_output


# class Classifier(nn.Module):
#     def __init__(self, 
#                  review_emb_dim=None,
#                  layer_1_out=256,
#                  num_classes=2,
#                  config=None
#                 ):
#         super(Classifier, self).__init__()
#         self.config = config

#         # when using GAT
#         self.fc1 = nn.Linear(768, 512)
#         self.dropout = nn.Dropout(0.1)
#         self.fc2 = nn.Linear(512, 2)

#         self.activation = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, input):
#         # print(f'>> input: {input.size()}')
#         output = input
#         # output = self.activation(output)
#         output = self.fc1(output)
#         output = self.activation(output)
#         output = self.dropout(output)
#         output = self.fc2(output)
#         softmax_output = self.softmax(output)
#         raw_unnormalized_scores = self.log_softmax(output)

#         return raw_unnormalized_scores, softmax_output
