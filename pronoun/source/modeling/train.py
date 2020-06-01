import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from source.utility.checkpoint import saveModel
from collections import OrderedDict

def validModel(device, model, data, loss_fn):
    model.eval()
    losses = []
    for sample in data:
        text = Variable(sample[0].to(device))
        offset = Variable(sample[1].to(device))
        feature_a = Variable(sample[2].squeeze().to(device))
        feature_b = Variable(sample[3].squeeze().to(device))
        label = Variable(sample[4].squeeze().to(device))
        preds = model(text, offset, feature_a, feature_b)
        loss = loss_fn(preds, label)
        losses.append(loss.data.item())
    mean_loss = round(np.mean(losses),4)
    model.train()
    return mean_loss

def trainModel(device, model, train, valid, loss_fn, optimizer, scheduler, save, epochs, batch):
    checkpoint = np.Infinity
    for epoch in range(epochs):
        model.train()
        tq = tqdm(total=len(train) * batch, disable=False)
        losses = []
        for sample in train:
            text = Variable(sample[0].to(device))
            offset = Variable(sample[1].to(device))
            feature_a = Variable(sample[2].to(device))
            feature_b = Variable(sample[3].to(device))
            label = Variable(sample[4].squeeze().to(device))
            preds = model(text, offset, feature_a, feature_b)
            loss = loss_fn(preds, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())
            train_loss = round(np.mean(losses),4)
            tq.update(text.shape[0])
            tq.set_postfix(train_loss='{:.4f}'.format(train_loss))
        valid_loss = validModel(device, model, valid, loss_fn)
        postfix = OrderedDict()
        postfix['train_loss'] = '{:.4f}'.format(train_loss)
        postfix['valid_loss'] = '{:.4f}'.format(valid_loss)
        tq.set_postfix(**postfix)
        tq.close()
        if valid_loss < checkpoint:
            checkpoint = valid_loss
            saveModel(epoch, model, valid_loss, save + 'model.pth')
        if scheduler is not None:
            scheduler.step(valid_loss)
    return None