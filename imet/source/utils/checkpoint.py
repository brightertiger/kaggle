import torch

def saveModel(epoch, model, loss, path):
    results = {}
    results['epoch'] = epoch
    results['model_state_dict'] = model.state_dict()
    results['loss'] = loss
    torch.save(results, path)
    return None

def loadModel(model, path):
    results = torch.load(path)
    model.load_state_dict(results['model_state_dict'])
    loss = results['loss']
    print('Model Loaded:', 'Loss:', loss)
    return model, loss