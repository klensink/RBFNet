import torch
import torch.nn.init
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
# plt.ion()

class RBFnet(nn.Module):

    def __init__(self, centers, num_classes = 1):
        super(RBFnet, self).__init__()
        self.u = nn.Parameter(centers, requires_grad = True)
        self.K = centers.shape[0]
        self.num_classes = num_classes
        self.g = nn.Parameter(torch.ones(1, centers.shape[0])*10, requires_grad = True)
        self.W = nn.Linear(self.K, self.num_classes, bias=False)

    def radial_kernel(self, X, u, g):
        n, d = X.shape 
        k, d = u.shape

        # Radial distances thru broadcasting
        u = u.unsqueeze(0) # Add singelton in 1st channel (1 x k x d)
        X = X.unsqueeze(1) # Add singelton in 2nd channel (n x 1 x d)
        Z = torch.exp(-g*((X - u).norm(p=2, dim=2)**2)) # Compute kernel with broadcasting (n x k)

        return Z

    def forward(self, x):
        Z = self.radial_kernel(x, self.u, self.g)
        x = self.W(Z) # ZW + b

        return x

def plot_state(model, x, x_test, count, title=None):
    preds = model(x_test)
    plt.figure()
    plt.plot(x.numpy(), y.detach().numpy(), 'kx--')
    plt.plot(x_test.numpy(), preds.detach().numpy(), 'b')
    c = model.u.detach().numpy()
    plt.plot(c, np.zeros_like(c), 'kx')
    plt.ylim([0, 5])

    if title is not None:
        plt.title(title + ' %6d' % count)
    plt.savefig('figs/%04d.png' % count)
    plt.close()

    return count + 1

# Training data 
# Create 5 gaussians
g_centers = [1, 3, 5, 7, 9]
g_weights = [1, 4, 1, 4, 1]
g_decays  = [1, 1, 1, 2, 1]

# N = 1024
# N = 1024*8
N = 16
x = np.linspace(0, 10, N).astype(np.float32)
Y = np.zeros((len(g_centers), N)).astype(np.float32)
for i,(c, w, d) in enumerate(zip(g_centers, g_weights, g_decays)):
    Y[i] = w*np.exp(-d*np.abs(x - c)**2)

[plt.plot(x, yi) for yi in Y]
y = np.sum(Y, 0)
x = torch.tensor(x).unsqueeze(1)
y = Variable(torch.tensor(y).unsqueeze(1), requires_grad = False)

# Test data
x_test = np.linspace(0, 10, 512).astype(np.float32)
x_test = torch.tensor(x_test).unsqueeze(1)

# Create model
cycles = 1
training_epochs = 200

# Gaussian Centers
cluster_centers = x
# cluster_centers = torch.rand(512,1)*10
# cluster_centers = torch.tensor([4., 6.]).unsqueeze(1)

model = RBFnet(cluster_centers)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
    [p[1] for p in model.named_parameters() if p[0] == 'W.weight' or p[0] == 'W.bias'],
    lr=1e-1, 
    momentum=0.1
)

# Train weights with fixed decay rates and cluster centers
for epoch in range(training_epochs):
    optimizer.zero_grad()               
    preds = model(x)                    
    loss = criterion(preds, y)     
    loss.backward()                     
    optimizer.step()                    
    print("[Epoch: %2d, Weights cost =      %f" % (epoch + 1, loss))
    count = plot_state(model, x, x_test, epoch, title='W')

# plt.show()
