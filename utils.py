import torch

from models import LSTM_Cartpole, FC_Cartpole

def save_policy_network(filepath, policy_net, training_info = None):
    checkpoint = {
        'state_dict': policy_net.state_dict(),
        'hyperparameters': policy_net.get_hyperparameters(),
        'class_name' : policy_net.__class__.__name__ 
    }
    if training_info is not None:
        checkpoint['training_info'] = training_info
    torch.save(checkpoint, filepath)

def load_policy_network(filepath):
    checkpoint = torch.load(filepath)
    model_class = globals()[checkpoint['class_name']]
    model = model_class(**checkpoint['hyperparameters'])
    model.load_state_dict(checkpoint['state_dict'])

    # training_info = checkpoint.get('training_info', None)
    
    return model