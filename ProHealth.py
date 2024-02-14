import math
import pickle as pickle
import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torchdiffeq import odeint_adjoint as odeint
import random
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0")
def load(emb_file_path):

    emb_dict = {}
    with open(emb_file_path,'r') as emb_file:        
        for i, line in enumerate(emb_file):
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split('\t')
                emb_dict[index] = np.array(emb.split()).astype(np.float32)
        
    return train_para, emb_dict  

emb_file_path = './data/emb.dat'
train_para, emb_dict = load(emb_file_path)
disease_emb=[]
for i in range(22117,len(emb_dict)):
    disease_emb.append(emb_dict[str(i)])
disease_emb=torch.tensor(disease_emb)
with open('./data/binary_train_codes_x.pkl', 'rb') as f0:
  binary_train_codes_x = pickle.load(f0)

with open('./data/binary_test_codes_x.pkl', 'rb') as f1:
  binary_test_codes_x = pickle.load(f1)

train_codes_y = np.load('./data/train_codes_y.npy')
train_visit_lens = np.load('./data/train_visit_lens.npy')

test_codes_y = np.load('./data/test_codes_y.npy')
test_visit_lens = np.load('./data/test_visit_lens.npy')
train_pids = np.load('./data/train_pids.npy')

test_pids = np.load('./data/test_pids.npy')
with open('./data/patient_time_duration_encoded.pkl', 'rb') as f80:
  patient_time_duration_encoded = pickle.load(f80)
 #Prepare padded hypergraphs for batching
def transform_and_pad_input(x):
  tempX = []
  for ele in x:
    tempX.append(torch.tensor(ele).to(torch.float32))
  x_padded = pad_sequence(tempX, batch_first=True, padding_value=0)
  return x_padded

trans_y_train = torch.tensor(train_codes_y)
trans_y_test = torch.tensor(test_codes_y)
padded_X_train = torch.transpose(transform_and_pad_input(binary_train_codes_x), 1, 2)
padded_X_test = torch.transpose(transform_and_pad_input(binary_test_codes_x), 1, 2)
class_num = train_codes_y.shape[1]

total_pids = list(train_pids) + list(test_pids)
cur_max = 0
for pid in total_pids:
  duration = patient_time_duration_encoded[pid]
  ts = [sum(duration[0:gap+1]) for gap in range(len(duration))]
  if cur_max < max(ts):
    cur_max = max(ts)
def prepare_temporal_encoding(H, pid, duration_dict, Dv):
  TE = []
  X_i_idx = torch.unique(torch.nonzero(H, as_tuple=True)[0])
  H_i = H[X_i_idx, :]
  for code in X_i_idx:
    TE_code = torch.zeros(Dv * 2)
    visits = torch.nonzero(H[code.item()]).tolist()
    temp = duration_dict[pid][:-1]
    code_duration = [sum(temp[0:gap+1]) for gap in range(len(temp))]
    visits.append([len(code_duration) - 1])
    pre_delta = [code_duration[visits[j][0]] - code_duration[visits[j - 1][0]] for j in range(1, len(visits))]
    delta = sum(pre_delta) / len(pre_delta)
    T_m = sum(code_duration)
    if T_m == 0:
      T_m += 1
    for k in range(len(TE_code)):
      if k < Dv:
        TE_code[k] = math.sin((k * delta) / (T_m * Dv))
      else:
        TE_code[k] = math.cos(((k - Dv) * delta) / (T_m * Dv))
    TE.append(TE_code)
  TE_i = torch.stack(TE)
  return TE_i


# In[ ]:


def load_prepared_TE(pids, te_dict):
  TE_list = []
  for pid in pids:
    one_patient = []
    patient_dict = te_dict[pid]
    for i, (k, v) in enumerate(patient_dict.items()):
      one_patient.append(torch.tensor(v))
    TE_list.append(torch.stack(one_patient))
  return TE_list
class ProHealth_Dataset(data.Dataset):
    def __init__(self, hyperG, data_label, pid, duration_dict, data_len, te_location, Dv):
        self.hyperG = hyperG
        self.data_label = data_label
        self.pid = pid
        self.data_len = data_len
        if te_location == None:
          TE_list = [prepare_temporal_encoding(hyperG[j], pid[j], duration_dict, Dv) for j in range(len(hyperG))]
          self.TE = pad_sequence(TE_list, batch_first=True, padding_value=0)
        else:
          with open(te_location, 'rb') as f250:
            TE_dict = pickle.load(f250)
          self.TE = pad_sequence(load_prepared_TE(pid, TE_dict), batch_first=True, padding_value=0)
 
    def __len__(self):
        return len(self.hyperG)
 
    def __getitem__(self, idx):
        return self.hyperG[idx], self.data_label[idx], self.pid[idx], self.TE[idx], self.data_len[idx]


# ### **Hierarchical Embedding for Medical Codes**

# In[ ]:


def glorot(tensor):
  if tensor is not None:
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    tensor.data.uniform_(-stdv, stdv)
class HierarchicalEmbedding(nn.Module):
    def __init__(self,embeddings):
        super(HierarchicalEmbedding, self).__init__()
        self.level_embeddings = nn.Embedding.from_pretrained(embeddings.to(torch.float32), freeze=False)
    def forward(self, input=None):
        embeddings_idx = [num for num in range(4880)]
        embeddings_idx=torch.tensor(embeddings_idx).to(device)
        #print(embeddings_idx)
        embeddings = self.level_embeddings(embeddings_idx)
        #print(embeddings)
        return embeddings # return: (code_num, one_code_dim * 4)
class Encoder(nn.Module):
  def __init__(self, code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer, nhead, personal_gate, hyperG_gate, PGatt_gate,X):
    super(Encoder, self).__init__()
    # Hierarchical embedding for medical codes
    self.hier_embed_layer = HierarchicalEmbedding(X)
    # Visit representation learning
    self.sigmoid = nn.Sigmoid()
    self.softmax = nn.Softmax()
    # Aggregate visit embeddings sequentially with attention
    self.temporal_edge_aggregator = nn.GRU(visit_dim, hdim, 1, batch_first=True)
    self.attention_context = nn.Linear(hdim, 1, bias=False)

  def forward(self, H, TE):
    #print(np.shape(X_G))
    #print(X_G)
    X=self.hier_embed_layer(None)
    visit_emb = torch.matmul(H.T.to(torch.float32), X)
    hidden_states, _ = self.temporal_edge_aggregator(visit_emb)
    alpha1 = self.softmax(torch.squeeze(self.attention_context(hidden_states), 1))
    h = torch.sum(torch.matmul(torch.diag(alpha1), hidden_states), 0)
    return h
class GRUODECell_Autonomous(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bias        = bias

        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, t, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.
        Args:
            t        time
            h        hidden state (current)
        Returns:
            Updated h
        """
        x = torch.zeros_like(h)
        z = torch.sigmoid(x + self.lin_hz(h))
        n = torch.tanh(x + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh

class ODEFunc(nn.Module):
  def __init__(self, hdim, ode_hid):
    super().__init__()
    self.func = nn.Sequential(nn.Linear(hdim, ode_hid),
                              nn.Tanh(),
                              nn.Linear(ode_hid, hdim))
    for m in self.func.modules():
      if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)
        nn.init.constant_(m.bias, val=0)

  def forward(self, t, y):
    output = self.func(y)
    return output
import numpy as np
import pdb
import gzip
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import operator
import scipy.io as sio
import os.path
import pandas as pd
from sklearn.manifold import TSNE
import pickle as pickle
def q_x(x_0, t):

    noise = torch.randn_like(x_0)
 
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
 
    return (alphas_t * x_0 + alphas_1_m_t * noise)  
def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
   
    t = torch.tensor([t]).to(device)

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x, t)
  
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
   
    sample = mean + sigma_t * z

    return (sample)
def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt,cur_x):
   
    #cur_x = torch.randn(shape)
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
    return cur_x
class denoise(nn.Module):
  def __init__(self, hdim,n_steps,num_units,visit_dim):
    super(denoise, self).__init__()
    self.n_steps = n_steps 
    self.num_steps = n_steps  
    self.hdim=hdim
    self.visit_dim=visit_dim
    self.num_units=num_units
    self.betas = torch.linspace(-6,6,self.num_steps).to(device)  
    self.betas = torch.sigmoid(self.betas)*(0.5e-2 - 1e-5)+1e-5
    self.alphas = 1-self.betas
    self.alphas_prod = torch.cumprod(self.alphas,0)
    self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
    self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
    self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
    self.alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device),self.alphas_prod[:-1]],0)
    assert self.alphas.shape==self.alphas_prod.shape==self.alphas_prod_p.shape==\
    self.alphas_bar_sqrt.shape==self.one_minus_alphas_bar_log.shape\
    ==self.one_minus_alphas_bar_sqrt.shape
    print("all the same shape",self.betas.shape)
    
    self.weight_cal=nn.Sequential(
                nn.Linear(2*self.visit_dim, self.visit_dim),
                nn.Tanh(),
                nn.Linear(self.visit_dim, 2,bias=False),
                nn.Softmax(dim=-1),
        )
    self.c_in=nn.Linear(self.hdim, self.visit_dim,bias=False)
    self.linears = nn.ModuleList(
            [
                nn.Linear(self.visit_dim, self.num_units),
                nn.ReLU(),
                nn.Linear(self.num_units, self.num_units),
                nn.ReLU(),
                nn.Linear(self.num_units, self.num_units),
                nn.ReLU(),
                nn.Linear(self.num_units, self.visit_dim),
            ]
        )
    self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.n_steps, self.num_units),
                nn.Embedding(self.n_steps, self.num_units),
                nn.Embedding(self.n_steps, self.num_units),
            ]
        )

  def forward(self, x, t):
    for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

    x = self.linears[-1](x)
    return x
class ODE_VAE_Decoder(nn.Module):
  def __init__(self, hdim, dist_dim, ode_hid, nclass, ODE_Func):
    super(ODE_VAE_Decoder, self).__init__()
    self.fc_mu = nn.Linear(hdim, dist_dim)
    self.fc_var = nn.Linear(hdim, dist_dim)
    self.fc_mu0 = nn.Linear(hdim, dist_dim)
    self.fc_var0 = nn.Linear(hdim, dist_dim)
    self.map_back = nn.Linear(2*dist_dim, hdim)
    self.relu = nn.ReLU()
    self.odefunc = ODE_Func
    self.final_layer = nn.Linear(hdim*2, nclass)
    self.softmax = nn.Softmax(dim=-1)
  def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    q = torch.distributions.Normal(mu, std)
    return q.rsample()

  def forward(self,z,timestamps):
    pred_z = odeint(func = self.odefunc, y0 = z, t = timestamps, method = 'rk4', options=dict(step_size=0.1))
    output = self.softmax(self.final_layer(pred_z))
    return output

class MLP_Decoder(nn.Module):
  def __init__(self, hdim, nclass):
    super(MLP_Decoder, self).__init__()
    self.final_layer = nn.Linear(hdim, nclass)
    self.softmax = nn.Softmax()

  def forward(self, h, timestamps):
    output = self.softmax(self.final_layer(h))
    return output, 0, 0


class ProHealth_VAE(nn.Module):
  def __init__(self, code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer, nhead, nclass, dist_dim, ode_hid, personal_gate, hyperG_gate, PGatt_gate, ODE_gate,disease_emb):
    super(ProHealth_VAE, self).__init__()
    self.encoder = Encoder(code_levels, single_dim, visit_dim, hdim, Dv, alpha_dim, gnn_dim, gnn_layer, nhead, personal_gate, hyperG_gate, PGatt_gate,disease_emb)
    self.ODE_Func = GRUODECell_Autonomous(hdim*2)
    self.decoder = ODE_VAE_Decoder(hdim, dist_dim, ode_hid, nclass, self.ODE_Func)
    self.softmax = nn.Softmax()
    self.num_step=100

    self.denoise = denoise(hdim*2,self.num_step,128,hdim*2)
    #self.visit_emb_pretrain=visit_emb
  def forward(self, Hs, TEs, timestamps, seq_lens,pids,duration_dict,past,truth):
    criterion = nn.BCELoss()
    

    h = torch.stack([self.encoder(Hs[ii][:, 0:int(seq_lens[ii])], TEs[ii]) for ii in range(len(Hs))])
    h0=torch.stack([self.encoder(Hs[ii][:, int(seq_lens[ii])-1:int(seq_lens[ii])], TEs[ii]) for ii in range(len(Hs))])
    
    mu = self.decoder.fc_mu(h)
    log_var = self.decoder.fc_var(h)
    z=self.decoder.reparameterize(mu, log_var)
    
    mu0 = self.decoder.fc_mu(h0)
    log_var0 = self.decoder.fc_var(h0)
    z0=self.decoder.reparameterize(mu0, log_var0)
    
    xi=torch.cat((h0,h),dim=1)#hdim*2
    zi=torch.cat((z0,z),dim=1)#dist_dim*2=hdim*2=512
    
  
    
    pred_z = odeint(func = self.decoder.odefunc, y0 = zi, t = timestamps, method = 'rk4', options=dict(step_size=0.1))
    pred_z = torch.swapaxes(pred_z, 0, 1)
    last_visits = []
    last_t=[]
    loss=0
    reconstruct_loss=0
    for i, traj in enumerate(pred_z):
      re_visit=[]
      duration = duration_dict[pids[i].item()]
      temp = [sum(duration[0:gap+1]) for gap in range(len(duration))]
      ts = [stamp / cur_max for stamp in temp]
      idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
      visit_lens = len(ts)
      last_visits.append(traj[idx[-1], :])
      last_t.append(traj[idx[-2], :])
      re_visit.append(traj[idx[0], :])
      visit=traj[idx[:], :]

      temp_loss=0
      for j in range(1,len(idx)):
        C_weight=self.denoise.weight_cal(torch.cat((traj[idx[j], :],self.denoise.c_in(traj[idx[j-1], :])),dim=0))#f(zt,zt-1)
        z_hat=torch.stack([C_weight[0]*traj[idx[j], :]+C_weight[1]*self.denoise.c_in(traj[idx[j-1], :])])
        t = torch.randint(0,self.num_step, size=(1 ,))

        t = t.unsqueeze(-1).to(device)
  
        a = self.denoise.alphas_bar_sqrt[t].to(device)
  
        aml = self.denoise.one_minus_alphas_bar_sqrt[t].to(device)
        e = torch.randn_like(z_hat).to(device)
        x = z_hat * a + e * aml
        output = self.denoise(x, t.squeeze(-1))
        temp_loss=temp_loss+(e - output).square().mean()
        z_hat=p_sample_loop(self.denoise, z_hat.shape, self.denoise.num_steps, self.denoise.betas, self.denoise.one_minus_alphas_bar_sqrt,z_hat)
        re_visit.append(z_hat[0])
      temp_reconstruct_loss=criterion(self.decoder.softmax(self.decoder.final_layer(torch.stack(re_visit[:-1]))),torch.swapaxes(past[i][:, 0:visit_lens-1],0,1))
      temp_reconstruct_loss=temp_reconstruct_loss+criterion(self.decoder.softmax(self.decoder.final_layer(re_visit[-1])),truth[i])
      reconstruct_loss += temp_reconstruct_loss/2
      loss=loss+(temp_loss/(len(idx)))
    loss=loss/len(pred_z)

    
    pred2=self.decoder(zi,timestamps)
    pred2 = torch.swapaxes(pred2, 0, 1)
    
    mug=torch.cat((mu0,mu),dim=1)
    log_varg=torch.cat((log_var0,log_var),dim=1)
    reconstruct_loss=(reconstruct_loss / len(pred_z))
    ELBO=torch.mean(-0.5 * torch.sum(1 + log_varg - mug ** 2 - log_varg.exp(), dim = 1))
    elbo=reconstruct_loss/ELBO
    ELBO = elbo*ELBO + reconstruct_loss
    
    return loss,0,pred2,mug,log_varg,ELBO

  def predict(self,Hs, TEs, timestamps, seq_lens,pids,duration_dict):
    h = torch.stack([self.encoder(Hs[ii][:, 0:int(seq_lens[ii])], TEs[ii]) for ii in range(len(Hs))])
    h0=torch.stack([self.encoder(Hs[ii][:, int(seq_lens[ii])-1:int(seq_lens[ii])], TEs[ii]) for ii in range(len(Hs))])
    
    mu = self.decoder.fc_mu(h)
    log_var = self.decoder.fc_var(h)
    z=self.decoder.reparameterize(mu, log_var)
    
    mu0 = self.decoder.fc_mu(h0)
    log_var0 = self.decoder.fc_var(h0)
    z0=self.decoder.reparameterize(mu0, log_var0)
    
    xi=torch.cat((h0,h),dim=1)#hdim*2
    zi=torch.cat((z0,z),dim=1)#dist_dim*2=hdim*2=512
    
    
    pred2=self.decoder(zi,timestamps)
    pred2 = torch.swapaxes(pred2, 0, 1)
    return 0,pred2
def IDCG(ground_truth, topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    idcg = 0
    for i in range(topn):
        idcg += ((2 ** t[i]) - 1) / math.log(i + 2, 2)
    return idcg


def nDCG(ranked_list, ground_truth, topn):
    dcg = 0
    idcg = IDCG(ground_truth, topn)
    for i in range(topn):
        idx = ranked_list[i]
        dcg += ((2 ** ground_truth[idx]) - 1)/ math.log(i + 2, 2)
    return dcg / idcg


# In[ ]:


def evaluate_model(pred, label, k1, k2, k3, k4, k5, k6):
    # Below is for nDCG
    ks = [k1, k2, k3, k4, k5, k6]
    y_pred = np.array(pred.cpu().detach().tolist())
    y_true_hot = np.array(label.cpu().detach().tolist())
    ndcg = np.zeros((len(ks), ))
    for i, topn in enumerate(ks):
        for pred2, true_hot in zip(y_pred, y_true_hot):
            ranked_list = np.flip(np.argsort(pred2))
            ndcg[i] += nDCG(ranked_list, true_hot, topn)
    n_list = ndcg / len(y_true_hot)
    metric_n_1 = n_list[0]; metric_n_2 = n_list[1]; metric_n_3 = n_list[2]; metric_n_4 = n_list[3]; metric_n_5 = n_list[4]; metric_n_6 = n_list[5]
    # Below is for precision and recall
    a = np.zeros((len(ks), )); r = np.zeros((len(ks), ))
    for pred2, true_hot in zip(y_pred, y_true_hot):
        pred2 = np.flip(np.argsort(pred2))
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred2[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / len(t)
    p_list = a / len(y_true_hot); r_list = r / len(y_true_hot)
    metric_p_1 = p_list[0]; metric_p_2 = p_list[1]; metric_p_3 = p_list[2]; metric_p_4 = p_list[3]; metric_p_5 = p_list[4]; metric_p_6 = p_list[5]
    metric_r_1 = r_list[0]; metric_r_2 = r_list[1]; metric_r_3 = r_list[2]; metric_r_4 = r_list[3]; metric_r_5 = r_list[4]; metric_r_6 = r_list[5]
    return metric_p_1, metric_r_1, metric_n_1, metric_p_2, metric_r_2, metric_n_2, metric_p_3, metric_r_3, metric_n_3, metric_p_4, metric_r_4, metric_n_4, metric_p_5, metric_r_5, metric_n_5, metric_p_6, metric_r_6, metric_n_6


# ### **Loss Function**

# In[ ]:
def ProHealth_loss(pred, truth, past, pids, mu, log_var, duration_dict, timestamps, ode_gate, balance, cur_max,ELBO):
  criterion = nn.BCELoss()
  if not ode_gate:
    loss = criterion(pred, truth)
  else:
    last_visits = []
    for i, traj in enumerate(pred):
      duration = duration_dict[pids[i].item()]
      temp = [sum(duration[0:gap+1]) for gap in range(len(duration))]
      ts = [stamp / cur_max for stamp in temp]
      idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
      visit_lens = len(ts)
      last_visits.append(traj[idx[-1], :])
    last_visits = torch.stack(last_visits)
    #rint(last_visits)
    #rint(np.shape(last_visits))
    #rint(last_visits)
    #last_visits=(last_visits+mlp_pred)/2
    #rint(np.shape(last_visits))
    pred_loss = criterion(last_visits, truth)
    loss =  pred_loss + balance * ELBO
  return loss
def train(model, lrate, num_epoch, train_loader, test_loader, model_directory, ode_gate, duration_dict, early_stop_range, balance, cur_max):
  model.train()
  losses=[]
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lrate)
  best_metric_r1=0
  test_loss_per_epoch = []; train_average_loss_per_epoch = []
  p1_list = []; p2_list = []; p3_list = []; p4_list = []; p5_list = []; p6_list = []
  r1_list = []; r2_list = []; r3_list = []; r4_list = []; r5_list = []; r6_list = []
  n1_list = []; n2_list = []; n3_list = []; n4_list = []; n5_list = []; n6_list = []
  for epoch in range(num_epoch):
    one_epoch_train_loss = []
    one_epoch_diff_loss = []
    one_epoch_ode_loss = []
    for i, (hyperGs, labels, pids, TEs, seq_lens) in enumerate(train_loader):
      hyperGs = hyperGs.to(device); labels = labels.to(device); TEs = TEs.to(device)
      hyperGs_vae=[]
      labels_vae=[]
      TEs_vae=[]
      pids_vae=[]
      seq_lens_vae=[]
      for patient_num in range(len(labels)):
        if seq_lens[patient_num]>1:
          hyperGs_vae.append(hyperGs[patient_num])
          labels_vae.append(labels[patient_num])
          TEs_vae.append(TEs[patient_num])
          pids_vae.append(pids[patient_num])
          seq_lens_vae.append(seq_lens[patient_num])
        
      hyperGs_vae=torch.stack(hyperGs_vae).to(device)
      labels_vae=torch.stack(labels_vae).to(device)
      TEs_vae=torch.stack(TEs_vae).to(device)
      timestamps = []
      for pid in pids_vae:
        duration = duration_dict[pid.item()]
        timestamps += [sum(duration[0:gap+1]) for gap in range(len(duration))]
      temp = [stamp / cur_max for stamp in list(set(timestamps))]
      timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
      #pred, mu, log_var,mlp_pred = model.predict(hyperGs_vae, TEs_vae, timestamps, seq_lens_vae)
      loss3,pred1,pred2,mu,log_var,ELBO=model(hyperGs_vae, TEs_vae, timestamps, seq_lens_vae,pids_vae,duration_dict,hyperGs_vae,labels_vae.to(torch.float32))#denoise_loss
      loss2 = ProHealth_loss(pred2, labels_vae.to(torch.float32), hyperGs_vae,pids_vae, mu, log_var, duration_dict, timestamps, ode_gate, 0.25, cur_max,ELBO)#pred+ELBO+supervision
      #loss1 = ProHealth_loss(pred1, labels_vae.to(torch.float32), hyperGs_vae,pids_vae, mu, log_var, duration_dict, timestamps, False,0.25, cur_max,ELBO)#re
      #one_epoch_diff_loss.append(loss1.item())
      #one_epoch_ode_loss.append(loss2.item())
      #loss=loss1
      loss=loss3*0.01+loss2
      one_epoch_train_loss.append(loss.item())
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
      optimizer.step()
    train_average_loss_per_epoch.append(sum(one_epoch_train_loss) / len(one_epoch_train_loss))
    #print("diff:",sum(one_epoch_diff_loss)/len(one_epoch_diff_loss),"ode:",sum(one_epoch_ode_loss)/len(one_epoch_ode_loss))
    print('Epoch: [{}/{}], Average Loss: {}'.format(epoch+1, num_epoch, round(train_average_loss_per_epoch[-1], 9)))
    '''
    plt.plot(train_average_loss_per_epoch)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    '''
    model.eval()
    one_epoch_test_loss = []
    test_data_len = 0
    pred_list = []
    truth_list = []
    for (hyperGs, labels, pids, TEs, seq_lens) in test_loader:
      hyperGs = hyperGs.to(device); labels = labels.to(device); TEs = TEs.to(device)
      hyperGs_vae=[]
      labels_vae=[]
      TEs_vae=[]
      pids_vae=[]
      seq_lens_vae=[]
      for patient_num in range(len(labels)):
        if seq_lens[patient_num]>1:
          hyperGs_vae.append(hyperGs[patient_num])
          labels_vae.append(labels[patient_num])
          TEs_vae.append(TEs[patient_num])
          pids_vae.append(pids[patient_num])
          seq_lens_vae.append(seq_lens[patient_num])
      hyperGs_vae=torch.stack(hyperGs_vae).to(device)
      labels_vae=torch.stack(labels_vae).to(device)
      TEs_vae=torch.stack(TEs_vae).to(device)
      with torch.no_grad():
        timestamps = []
        for pid in pids_vae:
          duration = duration_dict[pid.item()]
          timestamps += [sum(duration[0:gap+1]) for gap in range(len(duration))]
        temp = [stamp / cur_max for stamp in list(set(timestamps))]
        timestamps = torch.tensor(temp).to(torch.float32).sort()[0]
        pred1,pred2= model.predict(hyperGs_vae, TEs_vae, timestamps, seq_lens_vae,pids_vae,duration_dict)
      test_data_len += len(pids_vae)
      truth_list.append(labels_vae)
      if ode_gate:
        for jj, traj in enumerate(pred2):
          duration = duration_dict[pids_vae[jj].item()]
          ts1 = [sum(duration[0:gap+1]) for gap in range(len(duration))]
          ts = [stamp / cur_max for stamp in ts1]
          idx = [(timestamps == m).nonzero(as_tuple=True)[0].item() for m in ts]
          pred_list.append(traj[idx[-1], :])
      else:
        pred_list.append(pred)
        
    pred = torch.vstack(pred_list)
    #print(np.shape(pred))
    #print(np.shape(mlp_pred))
    #pred=(pred+mlp_pred)/2
    #print(np.shape(pred))
    truth = torch.vstack(truth_list)
    metric_p1, metric_r1, metric_n1, metric_p2, metric_r2, metric_n2, metric_p3, metric_r3, metric_n3, metric_p4, metric_r4, metric_n4, metric_p5, metric_r5, metric_n5, metric_p6, metric_r6, metric_n6, = evaluate_model(pred, truth, 5, 10, 15, 20, 25, 30)
    p1_list.append(metric_p1); p2_list.append(metric_p2); p3_list.append(metric_p3); p4_list.append(metric_p4); p5_list.append(metric_p5); p6_list.append(metric_p6)
    r1_list.append(metric_r1); r2_list.append(metric_r2); r3_list.append(metric_r3); r4_list.append(metric_r4); r5_list.append(metric_r5); r6_list.append(metric_r6)
    n1_list.append(metric_n1); n2_list.append(metric_n2); n3_list.append(metric_n3); n4_list.append(metric_n4); n5_list.append(metric_n5); n6_list.append(metric_n6)
    if metric_r1 > best_metric_r1:
        best_metric_r1 = metric_r1
        best_index = len(r1_list)-1
        #torch.save(model.state_dict(), f'{model_directory}/ProHealth.pth')
    with open('./train.log', 'a') as f:
        print(f'Test Epoch {epoch+1}: {round(metric_r1, 9)}; {round(metric_r4, 9)}; {round(metric_r6, 9)}; {round(metric_n1, 9)}; {round(metric_n4, 9)}; {round(metric_n6, 9)}',file=f)
        print("best:", "metric_r1_list:",r1_list[best_index],"metric_r2_list:", r2_list[best_index],"metric_n1_list:",n1_list[best_index],"metric_n2_list:",n2_list[best_index],file=f)
    model.train()
  return p1_list, p2_list, p3_list, p4_list, p5_list, p6_list, r1_list, r2_list, r3_list, r4_list, r5_list, r6_list, n1_list, n2_list, n3_list, n4_list, n5_list, n6_list, test_loss_per_epoch, train_average_loss_per_epoch
model = ProHealth_VAE(0, 32, 768, 256, 8, 64, 256, 2, 8, class_num, 256, 128, False,False, False, True,disease_emb).to(device)
#print(f'Number of parameters of this model: {sum(param.numel() for param in model.parameters())}')
te_directory = None
training_data = ProHealth_Dataset(padded_X_train, trans_y_train, train_pids, patient_time_duration_encoded, train_visit_lens, te_directory, 32)
train_loader = DataLoader(training_data, batch_size=128, shuffle=True)
test_data = ProHealth_Dataset(padded_X_test, trans_y_test, test_pids, patient_time_duration_encoded, test_visit_lens, te_directory, 32)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)
model_directory = './model'
p1_list, p2_list, p3_list, p4_list, p5_list, p6_list, r1_list, r2_list, r3_list, r4_list, r5_list, r6_list, n1_list, n2_list, n3_list, n4_list, n5_list, n6_list, test_loss_per_epoch, train_average_loss_per_epoch = train(model, 0.0001, 500, train_loader, test_loader, model_directory, True, patient_time_duration_encoded, 10, 0.5, cur_max)