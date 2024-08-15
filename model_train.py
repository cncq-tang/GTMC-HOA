import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from model.View_Block import ViewBlock
from GTMC_HOA import GTMC_HOA
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normal(dataset, args):
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims

    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]
    dataset.postprocessing(test_index, addNoise=False)
    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)

    view_blocks = [ViewBlock(v_num, dims[v_num][0], args.comm_feature_dim) for v_num in range(num_views)]
    model = GTMC_HOA(view_blocks, args.comm_feature_dim, num_classes, args.lambda_epochs, args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    model.to(device)
    model.train()
    for epoch in range(1, args.epochs + 1):
        print(f'====> {epoch}')
        for X, Y, indexes in train_loader:
            for v_num in range(len(X)):
                X[v_num] = X[v_num].to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            evidences, evidence_a, loss = model(X, Y, epoch)
            loss.backward()
            optimizer.step()

    model.eval()
    num_correct, num_sample = 0, 0
    for X, Y, indexes in test_loader:
        for v_num in range(len(X)):
            X[v_num] = X[v_num].to(device)
        Y = Y.to(device)
        with torch.no_grad():
            evidences, evidence_a, loss = model(X, Y, args.epochs + 1)
            _, Y_pre = torch.max(evidence_a, dim=1)
            num_correct += (Y_pre == Y).sum().item()
            num_sample += Y.shape[0]
    return num_correct / num_sample


def conflict(dataset, args):
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims

    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]
    dataset.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=0.1, addConflict=True, ratio_conflict=0.4)
    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)

    view_blocks = [ViewBlock(v_num, dims[v_num][0], args.comm_feature_dim) for v_num in range(num_views)]
    model = GTMC_HOA(view_blocks, args.comm_feature_dim, num_classes, args.lambda_epochs, args, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    model.to(device)
    model.train()
    for epoch in range(1, args.epochs + 1):
        print(f'====> {epoch}')
        for X, Y, indexes in train_loader:
            for v_num in range(len(X)):
                X[v_num] = X[v_num].to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            evidences, evidence_a, loss = model(X, Y, epoch)
            loss.backward()
            optimizer.step()

    model.eval()
    num_correct, num_sample = 0, 0
    for X, Y, indexes in test_loader:
        for v_num in range(len(X)):
            X[v_num] = X[v_num].to(device)
        Y = Y.to(device)
        with torch.no_grad():
            evidences, evidence_a, loss = model(X, Y, args.epochs + 1)
            _, Y_pre = torch.max(evidence_a, dim=1)
            num_correct += (Y_pre == Y).sum().item()
            num_sample += Y.shape[0]
    return num_correct / num_sample
