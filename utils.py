import torch

from models import CartpolePolicy

def save_policy_network(filepath, policy_net, training_info = None):
    checkpoint = {
        'state_dict': policy_net.state_dict(),
        'child_net_type': type(policy_net.model).__name__,
        'child_net_params': policy_net.model.get_hyperparameters() 
    }
    if training_info is not None:
        checkpoint['training_info'] = training_info
    torch.save(checkpoint, filepath)

def load_policy_network(filepath):
    checkpoint = torch.load(filepath)
    
    # Create the child network using the type and hyperparameters
    ChildNetClass = globals()[checkpoint['child_net_type']]
    child_net = ChildNetClass(**checkpoint['child_net_params'])

    # Create the policy network with the child network
    policy_net = CartpolePolicy(child_net)
    policy_net.load_state_dict(checkpoint['state_dict'])

    training_info = checkpoint.get('training_info', None)
    
    return policy_net