import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = nn.Sequential(
        nn.Linear(dim, hidden_dim), 
        norm(hidden_dim), 
        nn.ReLU(), 
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    return nn.Sequential(
        nn.Residual(modules),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
    ]
    for i in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(
        *modules
    )
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_val, batch_size, correct_count, sample_count = 0., 0, 0., 0
    if opt is not None:
        model.train()
    else:
        model.eval()
    loss_func = nn.SoftmaxLoss()
    for X, y in dataloader:
        batch_size += 1
        sample_count += y.shape[0]
        if opt is not None:
            opt.reset_grad()
        out = model(X)
        loss = loss_func(out, y)
        if opt is not None:
            loss.backward()
            opt.step()
        correct_count += (out.numpy().argmax(axis=1) == y.numpy()).sum()
        loss_val += loss.numpy()
    return (1 - correct_count / sample_count), loss_val / batch_size
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    train_dataset = ndl.data.MNISTDataset(
        data_dir + '/train-images-idx3-ubyte.gz',
        data_dir + '/train-labels-idx1-ubyte.gz'
    )
    test_dataset = ndl.data.MNISTDataset(
        data_dir + '/t10k-images-idx3-ubyte.gz',
        data_dir + '/t10k-labels-idx1-ubyte.gz'
    )
    train_loader = ndl.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_dataset, batch_size, shuffle=True)
    model = MLPResNet(28 * 28, hidden_dim, num_classes=10)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)
        test_err, test_loss = epoch(test_loader, model)
    return (train_err, train_loss, test_err, test_loss)



if __name__ == "__main__":
    train_mnist(data_dir="../data")
