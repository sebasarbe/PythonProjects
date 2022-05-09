#For data flair implementation
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

#For NN implementation
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

#Write csv file
import csv

#Measure time
import time

#Calculate log for hyperparameters search
import math

#Import location
directory = os.getcwd()
path = directory[:-4]+"news.csv"
df = pd.read_csv(path)

#Code from https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/ and https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_simple_fullynet.py
#Csv file writer
pathOutput = directory[:-4]+"results.csv"
file =open(pathOutput,"w")
writer=csv.writer(file)
headers=["num_epochs", "num_Hidden_layers", "hidden_size", "training_time", "testing_time", "test_accuracy","learning_rate"]
writer.writerow(headers)


#print(df.shape)
#print(df.head())

#Get labels
labels=df.label
#print(labels.head())

#Create test and train sets
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words="english",max_df=0.7)

#fit and transform train and  transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy of sklearn: {round(score*100,2)}%')

#DataFlair - Build confusion matrix
#c=confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
#print(c)

#NN class
class NN(nn.Module):
    def __init__(self, input_size, num_classes,num_hidden_layers,hidden_size,dropout_layer,dropout_layerRatio):
        super(NN, self).__init__()
        # The first linear layer take input_size nodes to hidden_size nodes.
        # The following num_hidden_layers linear layers take hidden_size nodes to hidden_size nodes.
        # and our second linear layer takes 50 to the num_classes we have.
        #If dropout_layer == True adds a dropout layer for training porpuses.
        self.net = nn.Sequential()
        self.net.append(nn.Linear(input_size, hidden_size))
        for i in range(num_hidden_layers):
            self.net.append(nn.ReLU())
            self.net.append(nn.Linear(hidden_size, hidden_size))
        self.net.append(nn.Linear(hidden_size, num_classes))
        if (dropout_layer):
            self.net.append(nn.Dropout(p=dropout_layerRatio))


    def forward(self, x):
        return self.net(x)


# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Creates a custom dataset class
class wordDataset(Dataset):
    def __init__(self,xdata,ydata):
        self.x= torch.from_numpy(xdata)
        self.y = torch.from_numpy(ydata)
        self.n_samples =xdata.shape[0]

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.n_samples

#Checks the acurracy of the model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x.float())
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

#Prints in the csv file the desired information
def printLineInCSV(pWriter,num_epochs,num_hidden_Layers,hidden_size,trainingTime,testingTime,testAcurracy,learningR):
    n=[]
    n.append(num_epochs)
    n.append(num_hidden_Layers)
    n.append(hidden_size)
    n.append(trainingTime)
    n.append(testingTime)
    n.append(testAcurracy)
    n.append(learningR)
    pWriter.writerow(n)

#Creates train and test loader for the NN
y_trainN= np.array([1 if y == "REAL" else 0 for y in y_train])
y_testN= np.array([1 if y == "REAL" else 0 for y in y_test])

#Hiperparameters
input_size = tfidf_train.shape[1]
num_classes = 2
batch_size = 64
dropout_layer=True
dropout_layerRatio=0.5


#Initialices custom datasets
trainDataSet=wordDataset(np.array(tfidf_train.toarray()),y_trainN)
testDataSet=wordDataset(np.array(tfidf_test.toarray()),y_testN)

#Define n_iterations per epoch
n_iterations=len(trainDataSet)/batch_size

#initialices dataloaders to iterate the data
train_loader = DataLoader(dataset=trainDataSet, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=testDataSet, batch_size=batch_size, shuffle=True)

#Define hyperparameters range

num_epochsI=10
num_epochsF=50

num_hidden_LayersF=10

hidden_sizeI=5
hidden_sizeF=50

learning_rateI = 0.0001
learning_rateF = 0.01

# Number of models to test hyperparameters
for numM in range(301):

    # Estimate the random number of epochs to test in this iteration
    num_epochs = round(10 ** np.random.uniform(math.log(num_epochsI, 10), math.log(num_epochsF, 10)))

    # Estimate the random number of hidden layers to test in this iteration
    num_hidden_Layers = np.where(np.random.multinomial(1,[1/(num_hidden_LayersF+1)]*(num_hidden_LayersF+1))==1)[0][0]

    # Estimate the number of hidden size of each layer to test in this iteration
    hidden_size = round(10 ** np.random.uniform(math.log(hidden_sizeI, 10), math.log(hidden_sizeF, 10)))

    # Estimate the random learning rate to test in this iteration
    learning_rate = 10 ** np.random.uniform(math.log(learning_rateI, 10), math.log(learning_rateF, 10))

    # Initialize network
    model = NN(input_size=input_size, num_classes=num_classes, num_hidden_layers=num_hidden_Layers,
               hidden_size=hidden_size,
               dropout_layer=dropout_layer, dropout_layerRatio=dropout_layerRatio).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainingTime = 0
    testingTime = 0

    start = time.time()
    # Training loop
    for epoch in range(num_epochs):
        for i, (inputs, pLabels) in enumerate(train_loader):
            # Puts the tensors in the device (GPU)
            inputs = inputs.to(device=device)
            pLabels = pLabels.to(device=device)
            # forward
            scores = model(inputs.float())
            loss = criterion(scores, pLabels)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

    trainingTime = time.time() - start

    # Checks acurracy
    start = time.time()
    testAccuracy = check_accuracy(test_loader, model).numpy() * 100
    testingTime = time.time() - start
    printLineInCSV(writer, num_epochs, num_hidden_Layers, hidden_size, trainingTime, testingTime, testAccuracy,learning_rate)
    print(
        f"{numM}. num_epochs:{num_epochs},num_hidden:{num_hidden_Layers}, hidden_size:{hidden_size},learning_rate:{learning_rate}, trainingTime:{trainingTime}, testingTime:{testingTime}, testAccuracy:{testAccuracy}")



file.close()
