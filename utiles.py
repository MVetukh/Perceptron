import json

def read_param():
    with open("config.json", 'r') as f:
        result = json.load(f)
        batch_size = result['grad_parameters']['batch_size']
        epochs = result['grad_parameters']['epochs']
        learning_rate = result['grad_parameters']['learning_rate']
    return batch_size, epochs, learning_rate

