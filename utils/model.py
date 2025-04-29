import torch
import torch.nn as nn
class model_arc(nn.Module):
          def __init__(self,inp_siz):
                  super(model_arc,self).__init__()
                  self.con1=nn.Conv2d(in_channels=inp_siz,out_channels=16,kernel_size=3,stride=1,padding=1)
                  self.b1=nn.BatchNorm2d(16) 

                  self.con2=nn.Conv2d(16,out_channels=32,kernel_size=3,stride=1,padding=1)
                  self.b2=nn.BatchNorm2d(32) 
                  
                  self.con3=nn.Conv2d(32,out_channels=64,kernel_size=3,stride=1,padding=1)
                  self.b3=nn.BatchNorm2d(64) 

                  self.con4=nn.Conv2d(64,out_channels=128,kernel_size=3,stride=1,padding=1)
                  self.b4=nn.BatchNorm2d(128)

                  self.maxp=nn.MaxPool2d(kernel_size=4,stride=2) 
                  self.relu=nn.ReLU()
              
                 
                  self.d3=nn.Dropout2d(p=0.5)
                  self.d4=nn.Dropout(p=0.5)

                  self.l1=nn.Linear(128*37*37,512)
                  self.l2=nn.Linear(512,1)
          def forward(self,inp):
                  x=self.relu(self.b1(self.con1(inp))) # 400,16,156,156
                  x=self.relu((self.b2(self.con2(x)))) # 400,32,156,156
                  x=self.d3(self.relu((self.b3(self.con3(x))))) # 400,64,156,156
                  x=self.maxp(x)  # 400,64,77,77
                  x=self.relu(self.b4(self.con4(x)))
                  x=self.maxp(x)  # 400,128,37,37

                  x=self.d4(self.relu(self.l1(torch.flatten(x,1))))
                  out=self.l2(x)

                  return out