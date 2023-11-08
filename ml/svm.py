import torch
import torch.nn as nn
import torch.optim as optim

def train(X, Y, model, args):
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)

    model = nn.Linear(args.input_size, args.output_size)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lam)

    model.train()
    for epoch in range(args.epoch):
        perm = torch.randperm(len(Y))
        sum_loss = 0

        for i in range(len(perm)):
            x = X[perm[i]]
            y = Y[perm[i]]

            optimizer.zero_grad()
            output = model(x).squeeze()

            loss = torch.mean(torch.clamp(1 - y * output, min=0))

            loss.backward()
            w_grad = torch.clamp(model.weight.grad, min=-args.range, max=args.range)
            b_grad = torch.clamp(model.bias.grad, min=-args.range, max=args.range)

            # w_grad, b_grad = apply_ldp(w_grad, b_grad, args)

            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))