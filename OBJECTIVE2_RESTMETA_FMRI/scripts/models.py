"""A trained signed graph-shift network and an information-matched linear control."""
import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import read, CONFIG
from representations import vectorize

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpectralGNN(nn.Module):
    def __init__(self, n=116, hidden=16, dropout=.2):
        super().__init__()
        self.w0 = nn.Parameter(torch.empty(n, hidden))
        self.b0 = nn.Parameter(torch.zeros(hidden))
        self.w1 = nn.Parameter(torch.empty(hidden, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w0); nn.init.xavier_uniform_(self.w1)

    def forward(self, s, embedding=False):
        h1 = torch.relu(s @ self.w0 + self.b0)
        h2 = torch.relu((s @ self.dropout(h1)) @ self.w1 + self.b1)
        h = h2.mean(dim=1)
        out = self.head(self.dropout(h)).squeeze(-1)
        return (out, h) if embedding else out

def predict_gnn(model, s):
    model.eval()
    ps, hs = [], []
    with torch.no_grad():
        for start in range(0, len(s), 256):
            z,h = model(torch.as_tensor(s[start:start+256],dtype=torch.float32,device=DEVICE),embedding=True)
            ps.append(torch.sigmoid(z).cpu().numpy()); hs.append(h.cpu().numpy())
    return np.concatenate(ps).astype(float), np.concatenate(hs)

def train_gnn(s, y, random_seed, decay, epochs, validation_s=None, snapshots=()):
    cfg=read(CONFIG)['models']
    torch.manual_seed(random_seed); torch.cuda.manual_seed_all(random_seed)
    model=SpectralGNN(s.shape[-1],cfg['hidden'],cfg['dropout']).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=decay)
    x=torch.as_tensor(s, dtype=torch.float32,device=DEVICE)
    target=torch.as_tensor(y,dtype=torch.float32,device=DEVICE)
    criterion=nn.BCEWithLogitsLoss()
    initial=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
    values={}; losses=[]
    for epoch in range(1,epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss=criterion(model(x),target)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip']); optimizer.step()
        if epoch in snapshots:
            values[epoch]=predict_gnn(model,validation_s)[0]
        if epoch in [1,epochs]: losses.append(float(loss.detach().cpu()))
    delta=float(sum(torch.sum((v.detach().cpu()-initial[k])**2).item() for k,v in model.state_dict().items())**.5)
    return model,values,{'seed':random_seed,'epochs':epochs,'weight_decay':decay,'training_loss_first_last':losses,'weight_update_l2':delta}

def train_lr(s,y,C,random_seed):
    scaler=StandardScaler().fit(vectorize(s))
    x=scaler.transform(vectorize(s))
    model=LogisticRegression(C=C,solver='lbfgs',max_iter=3000,tol=1e-6,random_state=random_seed)
    model.fit(x,y)
    if np.any(model.n_iter_>=model.max_iter):
        raise RuntimeError('LR convergence failure; do not silently report unconverged fit')
    return model,scaler

def predict_lr(model,scaler,s):
    return model.predict_proba(scaler.transform(vectorize(s)))[:,1]
