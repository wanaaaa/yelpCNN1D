# https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
from functionClass import *
from gensim.models import Word2Vec
import torch
import torch.optim as optim

device = 'cuda'

rateReviewTrainList, rateReviewTestList, maxListCount = dataRead()

xyDataLoader = DataLoaderFun(rateReviewTrainList, maxListCount, batchSize=2500)

textCNNmodel = trainFun(xyDataLoader, maxListCount, epochs=20)

# textCNNmodel = TextCnn(maxListCount).cuda(device=device)
textCNNmodel = TextCnn(maxListCount).cpu()
textCNNmodel.load_state_dict(torch.load('traindTextCNNmodel.model'))
textCNNmodel.eval()
# ================================================
# ================================================
# ================================================


xyTestDataLoader = DataLoaderFun(rateReviewTestList, maxListCount, batchSize=1)
for epoch in range(1):
    # print("num of epochs->", epoch)
    for step, batch in enumerate(xyTestDataLoader):
        x_test, y_test = tuple(t.to('cpu') for t in batch)
        y_pridict = textCNNmodel(x_test)

        print("y_pridict->", y_pridict, 'y_test->',  y_test)
        # break

torch.cuda.empty_cache()