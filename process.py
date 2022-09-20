import os
import sys
import torch.nn as nn
import torch.optim as optim
from functions import (
    load_mnist_data, determine_device, run_model,
    generate_file_name, save_results, save_state)
from models import MLP2, MLP3


def process(dataset_name, model_name, criterion_name, gamma_1, gamma_2, gamma_3,
            hidden_units_1, hidden_units_2, hidden_units_3,
            epochs, batch_size, directory):
    """Trains a neural network model on a dataset and saves the resulting 
    model accuracy and model parameters to files
    
    Parameters
    ----------
    dataset_name: str
        'mnist' 
    model_name: str
        'mlp2' (two-layer perceptron) or 'mlp3' (three-layer perceptron)
    criterion_name: str
        'ce' (Cross Entropy loss) or 'mse' (Mean Squared Error loss)
    gamma_1: float
        the mean-field scaling parameter for the first layer
    gamma_2: float
        the mean-field scaling parameter for the second layer
    gamma_3: float
        the mean-field scaling parameter for the third layer
    hidden_units_1: int
        the number of nodes in the first layer
    hidden_units_2: int
        the number of nodes in the second layer
    hidden_units_3: int
        the number of nodes in the third layer
    epochs: int
        number of times to iterate through the data set for training the model 
        and calculating accuracy
    batch_size: int
        the number of images per batch
    directory: str
        the local where accuracy results and model parameters are saved
        (requires folders 'results' and 'models')
    """

    # Information
    print("Dataset:    {}".format(dataset_name.upper()))
    print("Model:      {}".format(model_name.upper()))
    print("Criterion:  {}".format(criterion_name.upper()))
    if (gamma_3 is not None) and (hidden_units_3 is not None):
        print("Parameters: g_1={g_1}, g_2={g_2}, g_3={g_3}, h_1={h_1}, h_2={h_2}, h_3={h_3}, e={e}, b={b}".format(
            g_1=gamma_1, g_2=gamma_2, g_3=gamma_3, h_1=hidden_units_1, h_2=hidden_units_2, h_3=hidden_units_3, e=epochs, b=batch_size))
    else:
        print("Parameters: g_1={g_1}, g_2={g_2}, h_1={h_1}, h_2={h_2}, e={e}, b={b}".format(
            g_1=gamma_1, g_2=gamma_2, h_1=hidden_units_1, h_2=hidden_units_2, e=epochs, b=batch_size))

    # Determine device
    device = determine_device(do_print=True)
    
    # Load data
    if dataset_name.upper() == 'MNIST':
        train_loader, test_loader = load_mnist_data(batch_size=batch_size)
#    elif dataset_name.upper() == 'CIFAR10':
#        train_loader, test_loader = load_cifar_data(batch_size=batch_size)
    else:
        raise ValueError("Dataset '{0}' unknown".format(dataset_name))
        
    # Neural network model
    if model_name.upper() == 'MLP2':
        learning_rate_fc1 = 1.0 / ((hidden_units_1 ** (1 - 2 * gamma_1)) * (hidden_units_2 ** (3 - 2 * gamma_2)))
        learning_rate_fc2 = 1.0 / ((hidden_units_1 ** (1 - 2 * gamma_1)) * (hidden_units_2 ** (2 - 2 * gamma_2)))
        learning_rate_fc3 = 1.0 / (hidden_units_2 ** (2 - 2 * gamma_2))
        model = MLP2(hidden_units_1=hidden_units_1, hidden_units_2=hidden_units_2, gamma_1=gamma_1, gamma_2=gamma_2)
    elif model_name.upper() == 'MLP3':
        learning_rate_fc1 = 1.0 / ((hidden_units_1 ** (1 - 2 * gamma_1)) * (hidden_units_2 ** (2 - 2 * gamma_2)) * (hidden_units_3**(3 - 2 * gamma_3)))
        learning_rate_fc2 = 1.0 / ((hidden_units_1 ** (1 - 2 * gamma_1)) * (hidden_units_2 ** (1 - 2 * gamma_2)) * (hidden_units_3**(3 - 2 * gamma_3)))
        learning_rate_fc3 = 1.0 / ((hidden_units_2 ** (1 - 2 * gamma_2)) * (hidden_units_3 ** (2 - 2 * gamma_3)))
        learning_rate_fc4 = 1.0 / (hidden_units_3 ** (2 - 2 * gamma_3))
        model = MLP3(hidden_units_1=hidden_units_1, hidden_units_2=hidden_units_2, hidden_units_3=hidden_units_3, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3)
    else:
        raise ValueError("Model '{0}' unknown".format(dataset_name))
    model.to(device)
    
    # Criterion (loss function)
    if criterion_name.upper() == 'CE':
        criterion = nn.CrossEntropyLoss()
        do_encoding = False
    elif criterion_name.upper() == 'MSE':
        criterion = nn.MSELoss()
        do_encoding = True
    else:
        raise ValueError("Criterion '{0}' unknown".format(criterion_name))
    
    # Optimizer
    if model_name.upper() == 'MLP2':
        optimizer = optim.SGD([{"params": model.fc1.parameters(), "lr": learning_rate_fc1},
                              {"params": model.fc2.parameters(), "lr": learning_rate_fc2},
                              {"params": model.fc3.parameters(), "lr": learning_rate_fc3}], lr = 1.0)
    elif model_name.upper() == 'MLP3':
        optimizer = optim.SGD([{"params": model.fc1.parameters(), "lr": learning_rate_fc1},
                              {"params": model.fc2.parameters(), "lr": learning_rate_fc2},
                              {"params": model.fc3.parameters(), "lr": learning_rate_fc3},
                              {"params": model.fc4.parameters(), "lr": learning_rate_fc4}], lr = 1.0)
    else:
        raise ValueError("Model '{0}' unknown".format(dataset_name))
    model.to(device)

    # Run model
    results = run_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        do_encoding=do_encoding,
        epochs=epochs)

    # File name
    file_name = generate_file_name(
        dataset_name=dataset_name,
        model_name=model_name,
        criterion_name=criterion_name,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_3=gamma_3,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        hidden_units_3=hidden_units_3,
        epochs=epochs,
        batch_size=batch_size)
    
    # Save accuracy results
    results_directory = os.path.join(directory, 'results/')
    save_results(
        results=results,
        directory=results_directory,
        file_name=file_name)

    # Save model state
    models_directory = os.path.join(directory, 'models/')
    save_state(
        model=model,
        directory=models_directory,
        file_name=file_name)
    
    return


if __name__ == "__main__":
    
    #PARAMETERS TO RUN FROM COMMAND LINE
    # dataset_name = str(sys.argv[1])
    # model_name = str(sys.argv[2])
    # criterion_name = str(sys.argv[3])
    # gamma_1 = float(sys.argv[4])
    # gamma_2 = float(sys.argv[5])
    # gamma_3 = None if sys.argv[6] == 'None' else float(sys.argv[6])
    # hidden_units_1 = int(sys.argv[7])
    # hidden_units_2 = int(sys.argv[8])
    # hidden_units_3 = None if sys.argv[9] == 'None' else int(sys.argv[9])
    # epochs = int(sys.argv[10])
    # batch_size = int(sys.argv[11])
    # directory = str(sys.argv[12])
    
    #PARAMETERS TO RUN LOCALLY
    dataset_name = 'mnist'
    model_name = 'mlp2'
    criterion_name = 'mse'
    gamma_1 = 0.5
    gamma_2 = 0.8
    gamma_3 = None
    hidden_units_1 = 100
    hidden_units_2 = 50
    hidden_units_3 = None
    epochs = 5
    batch_size = 20
    directory = '/usr2/postdoc/jyu32/Documents/9_16_2022/'
    #directory = '/project/scalingnn/test/'
    #directory = '/usr2/postdoc/jyu32/Documents/01_22_2022/'

    process(
        dataset_name=dataset_name,
        model_name=model_name,
        criterion_name=criterion_name,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_3=gamma_3,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        hidden_units_3=hidden_units_3,
        epochs=epochs,
        batch_size=batch_size,
        directory=directory)
