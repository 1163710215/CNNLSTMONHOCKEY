""" How to use C3D network. """
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
def GetBatch(data,label,sampleNum=8,batchnum=8):
    for i in range(0,sampleNum//batchnum):
        low=i*batchnum
        x=data[low:low+batchnum]
        y=label[low:low+batchnum]
        yield x,y
class LSTMTagger(torch.nn.Module):
    def __init__(self):
        super(LSTMTagger,self).__init__()
        features=25088
        hidden=2048
        output=1
        layers=1
        lr=0.001
        self.LSTM=torch.nn.LSTM(features,hidden,layers,batch_first=True)
#         self.Linear=torch.nn.Linear(hidden,output)
        self.Linear=nn.Sequential(
            nn.Linear(hidden, hidden//4),
            nn.ReLU(True),
#             nn.Dropout(inplace=False),
            nn.Linear(hidden//4, hidden//4),
            nn.ReLU(True),
#             nn.Dropout(inplace=False),
            #nn.Linear(4096, 487),
            nn.Linear(hidden//4, output),
        )    
#         self.criteria=torch.nn.MSELoss()
        self.criteria = nn.BCEWithLogitsLoss()
        
        self.opt=torch.optim.Adam([{'params':self.LSTM.parameters()},
                                   {'params':self.Linear.parameters()}],lr)
    def forward(self,inputs):
        out,_=self.LSTM(inputs)
        out_last=out[:,-1,:]
        pred=self.Linear(out_last)
        si = nn.Sigmoid()
        
        return si(pred)

    
    def train(self,train_input,train_output,epochNum=100,batchNum=8,finalLoss=1e-5):
        '''
        对数据进行训练
        如果self.data/label已有数据(不是None)直接读取
        否则对默认路径:当前路径\\sample\\进行读取
        将数据按比例分成训练集和测试集
        训练直到完成所有epoch或达到finalLoss以下
        '''
#         if self.data is None:
#             self.loadData()
#         else:
#             data=self.data
#             label=self.label

#         sampleNum=self.sampleNum
#         num_test=int(0.2*sampleNum)
#         train_input = data[num_test:]
#         train_output = label[num_test:]
#         test_input = data[:num_test]
#         test_output = label[:num_test]
        trainNum=train_input.size(0)
#         if trainNum<batchNum:
#             raise Exception('样本太少，或减少batch size')

        self.LSTM.train()
        self.Linear.train()
        
        print('train')

        savedir='save'+os.sep
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        train_losses=[]
        for epoch in tqdm(range(epochNum)):
            train_loss=0
            test_loss=0
            for x,y in GetBatch(train_input,train_output,
                                trainNum,batchNum):
                
                self.opt.zero_grad()
                out,_=self.LSTM(x)
                out_last=out[:,-1,:]
                pred=self.Linear(out_last)
                loss=self.criteria(pred,y)
                
                loss.backward()
                self.opt.step()

                train_loss+=loss.item()

            train_loss/=trainNum//batchNum
#             print(train_loss)
            train_losses.append(train_loss)
            
#             #test loss
#             with torch.no_grad():
#                 out,_=self.LSTM(test_input)
#                 out_last=out[:,-1,:]
#                 pred=self.Linear(out_last)
#                 test_loss=torch.sqrt(self.criteria(pred,test_output))

#             print('epoch:{},train:{},test:{}'.format(
#                 epoch,train_loss,test_loss))

            if (epoch%20==0)or(train_loss<finalLoss):
                state = {'net1':self.LSTM.state_dict(),
                         'net2':self.Linear.state_dict(),
                         'optimizer':self.opt.state_dict()}
                saveName='2kClaloss{}.pth'.format(train_loss)
                torch.save(state,savedir+saveName)
                plt.figure()
                plt.plot(range(epoch+1),train_losses)
                plt.show()
#                 if test_loss<finalLoss:
#                     break