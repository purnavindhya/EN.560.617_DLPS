#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from torch.utils.data import DataLoader, TensorDataset

class DNN(nn.Module):
    '''
    input : nodes--- list
                form [input features, hiddenlayer1, hiddenlayer2,...., outputfeatures]
    '''
    def __init__(self, nodes):
        super(DNN, self).__init__()

        self.act = nn.ReLU()

        # if activation:
        #     self.act = activation()
        self.hidden_layers = nn.ModuleList()

        for node in range(len(nodes)-2):
            self.hidden_layers.append(nn.Linear(nodes[node], nodes[node+1]))
        self.output_layer = nn.Linear(nodes[-2] , nodes[-1])

        for layer in self.hidden_layers:
            std_dev = 2/ np.sqrt(layer.weight.shape[0] + layer.weight.shape[1])
            nn.init.normal_(layer.weight, mean=0, std=std_dev)
            nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        x = self.output_layer(x)
        return x
    
def dx_dy(u, x):
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]    

def get_f(x, u, nu):
    u_x = dx_dy(u, x)
    u_xx = dx_dy(u_x[:,0].reshape(-1,1), x)
    return u_xx - (1/nu)*u - (1/nu)*torch.exp(x)

def train(model, train_loader, train_f_loader, loss_fn, optimizer, nu, epochs):
    train_loss = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (x_u, u), x_f in zip(train_loader, train_f_loader) :
            u_pred = model(x_u)
            u_f_pred = model(x_f)
            f = get_f(x_f, u_f_pred, nu)
            loss = loss_fn(u_pred, u) + loss_fn(f, torch.zeros_like(f))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loader.append(epoch_loss/len(train_loader))
            
            # f = get_f(x_f, u_pred, nu)
            
            # u_x = dx_dy(u_pred, x_f)
            # u_xx = dx_dy(u_x[:,0].reshape(-1,1), x)
            # f = u_xx - (1/nu)*u_pred - (1/nu)*torch.exp(x)
            # loss = loss_fn(u_pred, u) + loss_fn(f, torch.zeros_like(f))
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # train_loss.append(loss.item())
            # epoch_loss += loss.item()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Epoch loss {epoch_loss.item()}')
    return train_loss

def test(model, test_loader, loss_fn, nu):
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for x, u in test_loader:
            u_pred = model(x)
            u_x = dx_dy(u_pred)
            u_xx = dx_dy(u_x)
            f = u_xx - (1/nu)*u_pred - (1/nu)*torch.exp(x)
            loss = loss_fn(u_pred, u) + loss_fn(f, torch.zeros_like(f))
            test_loss += loss.item()
    return test_loss / len(test_loader)

#%%

nu = 0.001
N_f = 300
Nmax = 3000

X_u_train = np.vstack([ -1, 1 ])
u_train = np.vstack([ 1, 0 ])

X_f_train = -1 + 2*lhs(1, N_f)
X_f_star = torch.linspace(-1,1,200).reshape(-1,1)

# Convert to torch tensors
X_f_train = torch.tensor(X_f_train, dtype=torch.float32)
X_f_star = torch.tensor(X_f_star, dtype=torch.float32)
X_u_train = torch.tensor(X_u_train, dtype=torch.float32)
u_train = torch.tensor(u_train, dtype=torch.float32)

#%%

model = DNN(torch.Tensor([1, 4, 4, 4, 4, 4, 4, 1]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

# dataset = TensorDataset( X_u_train, u_train )
train_loader = DataLoader(
    TensorDataset( X_u_train, u_train ),
    batch_size=32
)
train_f_loader = DataLoader(
    TensorDataset(X_f_train),
    batch_size=32
)

train_loss = train(model, train_loader, train_f_loader, loss_fn, optimizer, nu, 10)
test_loss = test(model, X_f_star, loss_fn, nu)
# %%
