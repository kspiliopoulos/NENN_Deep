import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_mnist_data(batch_size):
    """Return train and test data loaders for MNIST dataset
    
    Parameters
    ----------
    batch_size: int
        the number of images per batch
    """
    
    # Data normalization values from:
    # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
    transformation = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(
        'data/', train=True, transform=transformation, download=True)
    test_data = datasets.MNIST(
        'data/', train=False, transform=transformation, download=True)        
    train_loader = data.DataLoader(train_data, batch_size=batch_size)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader



def determine_device(do_print):
    """Determine whether to run the model on CPU or GPU
    
    Parameters
    ----------
    do_print: bool
        if True, print the device to be used
    """
    
    if torch.cuda.is_available():
        if do_print:
            print("Device:     GPU")
        device = torch.device("cuda")
    else:
        if do_print:
            print("Device:     CPU")
        device = torch.device("cpu")
 #   device = torch.device("cpu")
    return device


def run_model(model, optimizer, criterion, train_loader, test_loader, device,
              do_encoding, epochs):
    """Train model using specified optimizer, criterion, and training data
    
    Parameters
    ----------
    model: class inheriting nn.Module
        neural network model
    optimizer: from torch.optim
        optimization algorithm
    criterion: from torch.nn
        loss function
    train_loader: torch.utils.data.DataLoader
        training data loader
    test_loader: torch.utils.data.DataLoader
        test data loader
    device: torch.device("cuda") or torch.device("cpu")
        the device on which to run the model
    do_encoding: bool
        the MSE criterion requires one-hot encoding to calculate the loss
    epochs: int
        number of times to iterate through the data set for training the model 
        and calculating accuracy
    """
    
    # DataFrame to record accuracy results
    results = pd.DataFrame(
        None, index=range(epochs), columns=['Train','Test'], dtype=float)
    
    # Loop over epochs
    for epoch in range(epochs):

        # Train step
        train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            device=device,
            do_encoding=do_encoding)
        
        # Evaluation step
        model.eval()
        with torch.no_grad():
            
            # Train accuracy
            train_accuracy = calculate_accuracy(
                model=model,
                loader=train_loader,
                device=device)
            results.loc[epoch, 'Train'] = train_accuracy
            
            # Test accuracy
            test_accuracy = calculate_accuracy(
                model=model,
                loader=test_loader,
                device=device)
            results.loc[epoch, 'Test'] = test_accuracy
            
            # Print results
            msg = 'Epoch: {}, Train Accuracy = {:.2f}, Test Accuracy = {:.2f}'.format(
                epoch, train_accuracy, test_accuracy)
            print(msg)
        
    return results


def train(model, optimizer, criterion, train_loader, device, do_encoding):
    """Train model using specified optimizer, criterion, and training data
    
    Parameters
    ----------
    model: class inheriting nn.Module
        neural network model
    optimizer: from torch.optim
        optimization algorithm
    criterion: from torch.nn
        loss function
    train_loader: torch.utils.data.DataLoader
        training data loader
    device: torch.device("cuda") or torch.device("cpu")
        the device on which to run the model
    do_encoding: bool
        the MSE criterion requires one-hot encoding to calculate the loss
    """
    
    # Set to training mode
    model.train()
    
    # Loop over train data
    for batch in train_loader:
        
        # Set gradient to 0
        optimizer.zero_grad()
        
        # Load data
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Run the model
        output = model(inputs)

        # Evaluate criterion (i.e., calculate the loss)
        if do_encoding:
            targets_encoded = F.one_hot(targets, 10).float()
            loss = criterion(output, targets_encoded)
        else:
            loss = criterion(output, targets)

        # Back-propagation
        loss.backward()
        if hasattr(model, 'scale_learning_rates'):
            model.scale_learning_rates()
        optimizer.step()

    return


def calculate_accuracy(model, loader, device):
    """Calculate model accuracy for train or test data
    
    Parameters
    ----------
    model: class inheriting nn.Module
        neural network model
    loader: torch.utils.data.DataLoader
        train or test data loader
    device: torch.device("cuda") or torch.device("cpu")
        the device on which to run the model
    """
    
    # Image counters
    num_correct = 0
    num_attempt = 0
    
    # Loop over data
    for batch in loader:
    
        # Load data
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Run the model
        output = model(inputs)
    
        # Calculate accuracy for the batch
        predictions = output.max(dim=1, keepdim=True)[1]
        is_correct = predictions.eq(targets.view_as(predictions))
        num_correct += is_correct.sum().item()
        num_attempt += len(inputs)
    
    # Calculate test accuracy
    accuracy = num_correct / num_attempt
    return accuracy


def generate_file_name(dataset_name, model_name, criterion_name, gamma_1, gamma_2, gamma_3,
                       hidden_units_1, hidden_units_2, hidden_units_3, epochs, batch_size):
    """Returns a file name specifying parameters, e.g.,
        'mnist_mlp2_ce_gI06_gII05_hI1000_hII1000_e500_b20'
    
    Parameters
    ----------
    dataset_name: str
        'mnist' 
    model_name: str
        'mlp' or 'cnn'
    criterion_name: str
        'ce' (for Cross Entropy loss) or 'mse' (for Mean Squared Error loss)
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
    """

    parts = [dataset_name.lower(), model_name.lower(), criterion_name.lower()]
    g_1 = 'gI{0}'.format(str(gamma_1).replace('.',''))
    g_2 = 'gII{0}'.format(str(gamma_2).replace('.',''))
    parts += [g_1, g_2]
    if gamma_3 is not None:
        g_3 = 'gIII{0}'.format(str(gamma_3).replace('.',''))
        parts += [g_3]
    h_1 = 'hI{0}'.format(hidden_units_1)
    h_2 = 'hII{0}'.format(hidden_units_2)
    parts += [h_1, h_2]
    if hidden_units_3 is not None:
        h_3 = 'hIII{0}'.format(hidden_units_3)
        parts += [h_3]
    e = 'e{0}'.format(epochs)
    b = 'b{0}'.format(batch_size)
    parts += [e, b]
    file_name = '_'.join(parts)
    return file_name
    
    
def validate_directory(directory):
    """Checks if a directory exists, and creates it if it doesn't exist
    
    Parameters
    ----------
    directory: str
        the location to where the file will be saved
    """

    if not os.path.isdir(directory):
        os.makedirs(directory)
        msg = 'Created directory:  {directory}.'.format(
            directory=directory)
        print(msg)
    return
    
    
def save_results(results, directory, file_name):
    """Saves a csv file of the train and test accuracy by epoch
    
    Parameters
    ----------
    results: DataFrame
        index = epochs, columns = 'Train' and 'Test', data = accuracy
    directory: str
        the location to where the file will be saved
    file_name: str
        the name of the file
    """

    validate_directory(directory)
    extension = '.csv'
    file_path = os.path.join(directory, file_name) + extension
    results.to_csv(file_path)
    msg = 'Saved results to {file_path}.'.format(
        file_path=file_path)
    print(msg)
    return


def save_state(model, directory, file_name):
    """Saves a file of the model parameters
    
    Parameters
    ----------
    model: class inheriting nn.Module
        neural network model
    directory: str
        the location to where the file will be saved
    file_name: str
        the name of the file
    """

    validate_directory(directory)
    file_path = os.path.join(directory, file_name)
    state = model.state_dict()
    torch.save(obj=state, f=file_path)
    msg = 'Saved model to {file_path}.'.format(
        file_path=file_path)
    print(msg)
