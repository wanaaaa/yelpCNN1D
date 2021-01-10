import json
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from random import shuffle
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
import torch.optim as optim

w2vMyModel = Word2Vec.load('./w2vTrainedModel/w2vTrained.100th.10vec.model')

def dataRead():
    rateReviewList = []
    with open('review1000th.json') as file:
        porter_stemmer = PorterStemmer()
        lineCount = 0
        maxList = 0
        while True:
            line = file.readlines(1)
            if not line:
                break
            # if lineCount == 200:
            #     break
            jsonLine = json.loads(line[0])

            if jsonLine["stars"] <= 1:
                sentiNum = 0
            elif jsonLine["stars"] == 3:
                sentiNum = 1
            else:
                sentiNum = 2
            noStopWords = remove_stopwords(jsonLine['text'])
            stemWords = porter_stemmer.stem(noStopWords)
            tokenWords = simple_preprocess(stemWords, deacc=True)
            if len(tokenWords) > maxList:
                maxList = len(tokenWords)
            newDic = {}
            newDic['rate'] = sentiNum
            newDic['reviewTxt'] = tokenWords
            rateReviewList.append(newDic)

            lineCount = lineCount + 1
    shuffle(rateReviewList)

    splitPosi = int(len(rateReviewList)*0.8)
    rateReviewTrainList = rateReviewList[0:splitPosi]
    rareReviewTestList = rateReviewList[splitPosi:]
    return rateReviewTrainList, rareReviewTestList, maxList

def xDbListFun(xyDicList, maxListCt):
    for dic in xyDicList:
        stemList = dic['reviewTxt']
        if len(stemList) < maxListCt:
            for ele in range(maxListCt - len(stemList)):
                stemList.append('pad')

    reviewDbList = [ele['reviewTxt'] for ele in xyDicList]

    # print("training w2v.....")
    # w2v_model = Word2Vec(reviewDbList, size=10, workers=3, window=3, sg=1)
    # w2v_model.save('./w2vTrainedModel/w2vTrained.model')
    # print("end of training word2Vec---->>>")

    return reviewDbList

def xIndexDbListFun(xDbList, maxListCount):
    padIndex = []
    trainIndexList = [padIndex for i in range(maxListCount)]
    # w2vMyModel = Word2Vec.load('./w2vTrainedModel/w2vTrained.100th.10vec.model')

    xIndexDbLi = []
    for wordList in xDbList:
        nIndexList = []
        for word in wordList:
            nIndex = -777
            if word not in w2vMyModel.wv.vocab:
                nIndex = 0
            else:
                nIndex = w2vMyModel.wv.vocab[word].index
            nIndexList.append(nIndex)
        xIndexDbLi.append(nIndexList)

    return xIndexDbLi

def DataLoaderFun(xyDicList, maxListCount, batchSize):
    x_DbList = xDbListFun(xyDicList, maxListCount)
    xIndexDbList = xIndexDbListFun(x_DbList, maxListCount)
    yList = [ele['rate'] for ele in xyDicList]

    x_tensor, y_tensor = tuple( torch.tensor(ele) for ele in [xIndexDbList, yList]  )
    # print(y_tensor)

    xyTensor = TensorDataset(x_tensor, y_tensor)
    train_samper = RandomSampler(xyTensor)
    xyDataLoader = DataLoader(xyTensor, sampler=train_samper, batch_size=batchSize)

    return xyDataLoader

class TextCnn(nn.Module):
    def __init__(self, maxWordCt):
        super(TextCnn, self).__init__()
        self.maxWordCt = maxWordCt
        weights = w2vMyModel.wv
        self.embeddingDim = 10
        self.kernelSizes = [3, 4, 5]
        self.numFilters = [77, 77, 77]
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors),
                                                   padding_idx=w2vMyModel.wv.vocab['pad'].index)
        self.conv1List = nn.ModuleList([
            nn.Conv1d(in_channels=self.embeddingDim, out_channels=self.numFilters[i],
                kernel_size=self.kernelSizes[i]) for i in range(len(self.kernelSizes))
        ])

        # self.conv1 = nn.Conv1d(in_channels=self.embeddingDim, out_channels= self.numFilters,
        #                        kernel_size= self.kernelSize)
        self.fc = nn.Linear(np.sum(self.numFilters), 3)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, lineList):
        # print("maxWord====>", self.maxWordCt)
        x = self.embedding(lineList)  # [B, T, E]
        # print("after embedding-->", x.shape)
        x = x.permute(0,2, 1)
        # print("after permute-->", x.shape)
        x = [F.relu(con1d(x)) for con1d in self.conv1List]
        x = [F.max_pool1d(ele, kernel_size=ele.shape[2]) for ele in x]
        x = torch.cat([ele.squeeze(dim=2) for ele in x], dim=1)

        x = self.fc(self.dropout(x))
        return x

def trainFun(xyDataLoader, maxListCount, epochs):
    device = 'cuda'
    textCNNmodel = TextCnn(maxListCount).cuda(device=device)
    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(textCNNmodel.parameters(), lr=0.001)

    epochs = epochs
    for epoch in range(epochs):
        # print("num of epochs->", epoch)
        totalLoss = 0
        textCNNmodel.train()
        for step, batch in enumerate(xyDataLoader):
            x_train, y_labels = tuple(t.to(device) for t in batch)

            textCNNmodel.zero_grad()

            y_predict = textCNNmodel(x_train)
            # print(y_predict)
            loss = lossFunction(y_predict, y_labels)
            totalLoss += loss.item()
            loss.backward()

            optimizer.step()
            print("num of epochs->", epoch, " total loss->", totalLoss)

    torch.save(textCNNmodel.state_dict(), 'traindTextCNNmodel.model')

    return textCNNmodel