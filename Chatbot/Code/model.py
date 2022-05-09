import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel# Load the DistilBert tokenizer

class BERT_Arch(nn.Module):
    def __init__(self,k_classes):
        super(BERT_Arch, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # Set device cuda for GPU if it's available otherwise run on the CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")# Import the DistilBert pretrained model
        # dropout layer
        self.dropout = nn.Dropout(0.2)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k_classes)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)
        for param in self.bert.parameters():
            param.requires_grad = False


    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:, 0]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc3(x)
        # apply softmax activation
        x = self.softmax(x)
        return x
        # freeze all the parameters. This will prevent updating of model weights during fine-tuning.