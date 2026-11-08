import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

class CompactCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers=[]; previous=channels
        for width in (16,32,64):
            layers += [nn.Conv1d(previous,width,9,stride=2,padding=4),nn.BatchNorm1d(width),
                       nn.ReLU(),nn.MaxPool1d(2)]
            previous=width
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(64,1))
    def forward(self,x):
        return self.head(self.encoder(x)).squeeze(-1)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

def predict(model,x,device,batch=64):
    model.eval(); outputs=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):
            outputs.append(torch.sigmoid(model(torch.as_tensor(x[start:start+batch],device=device))).cpu().numpy())
    return np.concatenate(outputs)

def fit(x,y,vx,vy,seed,lr,weight_decay,epochs=40,patience=7,device='cpu'):
    seed_all(seed); model=CompactCNN(x.shape[1]).to(device)
    count=np.bincount(y.astype(int),minlength=2)
    if not count.all(): raise ValueError('Training requires both classes')
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(count[0]/count[1],dtype=torch.float32,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.tensor(y,dtype=torch.float32)),
                      batch_size=32,shuffle=True,generator=generator)
    history=[]; best=-float('inf'); state=None; wait=0
    for epoch in range(epochs):
        model.train(); losses=[]
        for bx,by in loader:
            optimizer.zero_grad(); loss=criterion(model(bx.to(device)),by.to(device))
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        vp=predict(model,vx,device); score=float(average_precision_score(vy,vp))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_auprc':score})
        if score>best:
            best=score; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
        if wait>=patience: break
    model.load_state_dict(state)
    return model,history,best
