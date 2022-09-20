import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP2(nn.Module):
    """Multi-layer perceptron with two hidden layers
    
    Attributes
    ----------
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    gamma_1: float
        the mean-field scaling parameter for the first hidden layer
    gamma_2: float
        the mean-field scaling parameter for the second hidden layer
    """
    
    def __init__(self, hidden_units_1, hidden_units_2, gamma_1, gamma_2):
        super(MLP2, self).__init__()
        
        # Parameters
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        
        # Layers
        self.fc1 = nn.Linear(28 * 28, hidden_units_1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        self.fc2 = nn.Linear(hidden_units_1, hidden_units_2)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)
        self.fc3 = nn.Linear(hidden_units_2, 10)
        nn.init.uniform_(self.fc3.weight, a=0.0, b=1.0)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        scaling_1 = self.hidden_units_1 ** (-self.gamma_1)
        x = scaling_1 * torch.sigmoid(self.fc1(x))
        scaling_2 = self.hidden_units_2**(-self.gamma_2)
        x = scaling_2 * torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class MLP3(nn.Module):
    """Multi-layer perceptron with three hidden layers
    
    Attributes
    ----------
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    hidden_units_3: int
        the number of nodes in the third hidden layer 
    gamma_1: float
        the mean-field scaling parameter for the first hidden layer
    gamma_2: float
        the mean-field scaling parameter for the second hidden layer
    gamma_3: float
        the mean-field scaling parameter for the third hidden layer 
    """
    
    def __init__(self, hidden_units_1, hidden_units_2, hidden_units_3, gamma_1, gamma_2, gamma_3):
        super(MLP3, self).__init__()
        
        # Parameters
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.hidden_units_3 = hidden_units_3
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.gamma_3 = gamma_3
        
        # Layers
        self.fc1 = nn.Linear(28 * 28, hidden_units_1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        self.fc2 = nn.Linear(hidden_units_1, hidden_units_2)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)
        self.fc3 = nn.Linear(hidden_units_2, hidden_units_3)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1.0)
        self.fc4 = nn.Linear(hidden_units_3, 10)
        nn.init.uniform_(self.fc4.weight, a=0.0, b=1.0)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        scaling_1 = self.hidden_units_1 ** (-self.gamma_1)
        x = scaling_1 * torch.sigmoid(self.fc1(x))
        scaling_2 = self.hidden_units_2**(-self.gamma_2)
        x = scaling_2 * torch.sigmoid(self.fc2(x))
        scaling_3 = self.hidden_units_3**(-self.gamma_3)
        x = scaling_3 * torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


