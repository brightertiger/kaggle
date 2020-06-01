import torch
import sys

def save_model(epoch, model, loss, path):
    results = {}
    results['epoch'] = epoch 
    results['model_state_dict'] = model.state_dict()
    results['loss'] = loss 
    torch.save(results, path)
    return None

def load_model(model, path):
    results = torch.load(path)
    model.load_state_dict(results['model_state_dict'])
    loss = results['loss']
    print('Model Loaded:', 'Loss:', loss)
    sys.stdout.flush()
    return model, loss