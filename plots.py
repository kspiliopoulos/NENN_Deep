import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from functions import generate_file_name, validate_directory


def load_accuracy_for_single_gamma(directory, dataset_name, model_name,
                                   criterion_name, gamma_1, gamma_2, gamma_3, 
                                   hidden_units_1, hidden_units_2, 
                                   hidden_units_3, epochs, batch_size):
    """Returns a DataFrame with test and train accuracy by epoch for single 
    gamma values
    
    Parameters
    ----------
    directory: str
        location of the data file
    dataset_name: str
        'mnist' 
    model_name: str
        'mlp2' or 'mlp3'
    criterion_name: str
        'ce' (for Cross Entropy loss) or 'mse' (for Mean Squared Error loss)
    gamma_1: float
        the mean-field scaling parameter for the first layer
    gamma_2: float
        the mean-field scaling parameter for the second layer
    gamma_3: float
        the mean-field scaling parameter for the third layer 
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    hidden_units_3: int
        the number of nodes in the third hidden layer
    epochs: int
        number of times to iterate through the data set for training the model 
        and calculating accuracy
    batch_size: int
        the number of images per batch
    """

    # Determine data file name
    if gamma_3 is None:
        fname = generate_file_name(
            dataset_name=dataset_name,
            model_name=model_name,
            criterion_name=criterion_name,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            gamma_3=None,
            hidden_units_1=hidden_units_1,
            hidden_units_2=hidden_units_2,
            hidden_units_3=None,
            epochs=epochs,
            batch_size=batch_size)
    else:
        fname = generate_file_name(
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

    
    # Create full path to data file, including extension
    results_folder = 'results/'
    path = os.path.join(directory, results_folder)
    path = os.path.join(path, fname) + '.csv'
    
    # Load data file
    data = pd.read_csv(path, index_col=0)

    return data


def load_all_accuracy(directory, dataset_name, model_name, criterion_name, 
                      gamma1_list, gamma2_list, gamma3_list, 
                      hidden_units_1, hidden_units_2, hidden_units_3, 
                      epochs, batch_size, is_test_accuracy):
    """Returns a DataFrame with either test or train accuracy by epoch for  
    lists of gamma values
    
    Parameters
    ----------
    directory: str
        location of the data file
    dataset_name: str
        'mnist' or 'cifar10'
    model_name: str
        'mlp' or 'cnn'
    criterion_name: str
        'ce' (for Cross Entropy loss) or 'mse' (for Mean Squared Error loss)
    gamma1_list: list of floats
        the mean-field scaling parameters for the first layer
    gamma2_list: list of floats
        the mean-field scaling parameters for the second layer
    gamma3_list: list of floats
        the mean-field scaling parameters for the third layer
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    hidden_units_3: int
        the number of nodes in the third hidden layer
    epochs: int
        number of times to iterate through the data set for training the model 
        and calculating accuracy
    batch_size: int
        the number of images per batch
    is_test_accuracy: bool
        True for test accuracy or False for train accuracy
    """

    column = 'Test' if is_test_accuracy else 'Train'
    
    # Dictionary to store data by different gamma_1
    dict_data = dict()

    # Iterate over list of gamma values and load accuracy data
    if gamma3_list==None:
        for gamma_1 in gamma1_list:
            for gamma_2 in gamma2_list:
                data = load_accuracy_for_single_gamma(
                    directory=directory,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    criterion_name=criterion_name,
                    gamma_1=gamma_1,
                    gamma_2=gamma_2,
                    gamma_3=None,
                    hidden_units_1=hidden_units_1,
                    hidden_units_2=hidden_units_2,
                    hidden_units_3=None,
                    epochs=epochs,
                    batch_size=batch_size)
                dict_data[(gamma_1,gamma_2)] = data[column]
    else:
        for gamma_1 in gamma1_list:
            for gamma_2 in gamma2_list:
                for gamma_3 in gamma3_list:
                    data = load_accuracy_for_single_gamma(
                        directory=directory,
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
                    dict_data[(gamma_1, gamma_2, gamma_3)] = data[column]
        
    # Concatenate accuracy data over gamma values
    results = pd.concat(dict_data, axis=1)
    
    return results



def run_3layer_accuracy_plots(dataset_name, model_name, criterion_name,
                              gamma1_list, gamma2_list, gamma3_list,
                              hidden_units_1, hidden_units_2, hidden_units_3, 
                              epochs, batch_size, is_test_accuracy, directory):
    """Plots and saves figures of test or train accuracy for lists of multiple 
    gamma values for Multi-layer perceptron with three hidden layers (MLP3)
    
    Parameters
    ----------
    dataset_name: str
        'mnist' 
    model_name: str
        'mlp3' 
    criterion_name: str
        'ce' (for Cross Entropy loss) or 'mse' (for Mean Squared Error loss)
    gamma1_list: list of floats
        the mean-field scaling parameters for the first layer 
    gamma2_list: list of floats
        the mean-field scaling parameters for the second layer 
    gamma3_list: list of floats
        the mean-field scaling parameters for the third layer 
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    hidden_units_3: int
        the number of nodes in the third hidden layer
    epochs: int
        number of times to iterate through the data set for training the model 
        and calculating accuracy
    batch_size: int
        the number of images per batch
    is_test_accuracy: bool
        True for test accuracy or False for train accuracy
    directory: str
        location of the data files and figures
    """

    # Load accuracy data
    data = load_all_accuracy(
        directory=directory,
        dataset_name=dataset_name,
        model_name=model_name,
        criterion_name=criterion_name,
        gamma1_list=gamma1_list,
        gamma2_list=gamma2_list,
        gamma3_list=gamma3_list,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        hidden_units_3=hidden_units_3,
        epochs=epochs,
        batch_size=batch_size,
        is_test_accuracy=is_test_accuracy) #.loc[1000:1500]
    
    line_styles = ['solid', 'dashed', 'dotted']
    colors = ['blue', 'green', 'red']
    
    # Create figures and plot data
    for gamma_1 in gamma1_list:
        
        fig = plt.figure(figsize=(20, 10))
        
        legend_labels = []
        for count_1, gamma_2 in enumerate(gamma2_list):

            for count_2, gamma_3 in enumerate(gamma3_list):
                legend_labels += [(gamma_2, gamma_3)]
                plt.plot(
                    data[(gamma_1, gamma_2, gamma_3)],
                    color=colors[count_1],
                    linestyle=line_styles[count_2])

        # Set title, label legend and x- and y-axes
        ax = fig.axes[0]
        ax.set_ylim([0,1])
        y_label = 'Test Accuracy' if is_test_accuracy else 'Train Accuracy'
        plt.ylabel(y_label)
        ax.set_title(y_label + ' for gamma_1={0}'.format(gamma_1))
        plt.legend(legend_labels, title='(gamma_2, gamma_3)', loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.xlabel('Number of Epochs')
    
        
        # Generate file name
        train_test = 'test' if is_test_accuracy else 'train'
        fname = 'plot_{dataset_name}_{model_name}_gI{gamma_1}_hI{hidden_1}_hII{hidden_2}_hIII{hidden_3}_e{epochs}_b{batch_size}_{train_test}'.format(
            dataset_name=dataset_name,
            model_name=model_name,
            gamma_1=gamma_1,
            hidden_1=hidden_units_1,
            hidden_2=hidden_units_2,
            hidden_3=hidden_units_3,
            epochs=epochs,
            batch_size=batch_size,
            train_test=train_test).replace('.', '')
    
        # Generate full path
        figures_directory = os.path.join(directory, 'figures/')
        validate_directory(directory=figures_directory)
        path = os.path.join(figures_directory, fname)
        
        # Save figure
        ax.figure.savefig(path + '.png', dpi=300, bbox_inches='tight')
        ax.figure.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
        msg = "Successfully saved to {fname}".format(fname=fname)
        print(msg)

    for gamma_2 in gamma2_list:
        
        fig = plt.figure(figsize=(20, 10))
        
        legend_labels = []
        for count_1, gamma_1 in enumerate(gamma1_list):

            for count_2, gamma_3 in enumerate(gamma3_list):
                legend_labels += [(gamma_1, gamma_3)]
                plt.plot(
                    data[(gamma_1, gamma_2, gamma_3)],
                    color=colors[count_1],
                    linestyle=line_styles[count_2])

        # Set title, label legend and x- and y-axes
        ax = fig.axes[0]
        ax.set_ylim([0,1])
        y_label = 'Test Accuracy' if is_test_accuracy else 'Train Accuracy'
        plt.ylabel(y_label)
        ax.set_title(y_label + ' for gamma_2={0}'.format(gamma_2))
        plt.legend(legend_labels, title='(gamma_1, gamma_3)', loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.xlabel('Number of Epochs')
    
        
        # Generate file name
        train_test = 'test' if is_test_accuracy else 'train'
        fname = 'plot_{dataset_name}_{model_name}_gII{gamma_2}_hI{hidden_1}_hII{hidden_2}_hIII{hidden_3}_e{epochs}_b{batch_size}_{train_test}'.format(
            dataset_name=dataset_name,
            model_name=model_name,
            gamma_2=gamma_2,
            hidden_1=hidden_units_1,
            hidden_2=hidden_units_2,
            hidden_3=hidden_units_3,
            epochs=epochs,
            batch_size=batch_size,
            train_test=train_test).replace('.', '')
    
        # Generate full path
        figures_directory = os.path.join(directory, 'figures/')
        validate_directory(directory=figures_directory)
        path = os.path.join(figures_directory, fname)
        
        # Save figure
        ax.figure.savefig(path + '.png', dpi=300, bbox_inches='tight')
        ax.figure.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
        msg = "Successfully saved to {fname}".format(fname=fname)
        print(msg)

    for gamma_3 in gamma3_list:
        
        fig = plt.figure(figsize=(20, 10))
        
        legend_labels = []
        for count_1, gamma_1 in enumerate(gamma1_list):

            for count_2, gamma_2 in enumerate(gamma2_list):
                legend_labels += [(gamma_1, gamma_2)]
                plt.plot(
                    data[(gamma_1, gamma_2, gamma_3)],
                    color=colors[count_1],
                    linestyle=line_styles[count_2])

        # Set title, label legend and x- and y-axes
        ax = fig.axes[0]
        ax.set_ylim([0,1])
        y_label = 'Test Accuracy' if is_test_accuracy else 'Train Accuracy'
        plt.ylabel(y_label)
        ax.set_title(y_label + ' for gamma_3={0}'.format(gamma_3))
        plt.legend(legend_labels, title='(gamma_1, gamma_2)', loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
        plt.xlabel('Number of Epochs')
    
        
        # Generate file name
        train_test = 'test' if is_test_accuracy else 'train'
        fname = 'plot_{dataset_name}_{model_name}_gIII{gamma_3}_hI{hidden_1}_hII{hidden_2}_hIII{hidden_3}_e{epochs}_b{batch_size}_{train_test}'.format(
            dataset_name=dataset_name,
            model_name=model_name,
            gamma_3=gamma_3,
            hidden_1=hidden_units_1,
            hidden_2=hidden_units_2,
            hidden_3=hidden_units_3,
            epochs=epochs,
            batch_size=batch_size,
            train_test=train_test).replace('.', '')
    
        # Generate full path
        figures_directory = os.path.join(directory, 'figures/')
        validate_directory(directory=figures_directory)
        path = os.path.join(figures_directory, fname)
        
        # Save figure
        ax.figure.savefig(path + '.png', dpi=300, bbox_inches='tight')
        ax.figure.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
        msg = "Successfully saved to {fname}".format(fname=fname)
        print(msg)
    return


def run_2layer_accuracy_plots(dataset_name, model_name, criterion_name,
                              gamma1_list, gamma2_list, hidden_units_1,
                              hidden_units_2, epochs, batch_size, 
                              is_test_accuracy, directory):
    """Plots and saves figures of test or train accuracy for lists of multiple 
    gamma values for Multi-layer perceptron with two hidden layers (MLP2)
    
    Parameters
    ----------
    dataset_name: str
        'mnist' 
    model_name: str
        'mlp2' 
    criterion_name: str
        'ce' (for Cross Entropy loss) or 'mse' (for Mean Squared Error loss)
    gamma1_list: list of floats
        the mean-field scaling parameters for the first layer 
    gamma2_list: list of floats
        the mean-field scaling parameters for the second layer 
    hidden_units_1: int
        the number of nodes in the first hidden layer
    hidden_units_2: int
        the number of nodes in the second hidden layer
    epochs: int
        number of times to iterate through the data set for training the model 
        and calculating accuracy
    batch_size: int
        the number of images per batch
    is_test_accuracy: bool
        True for test accuracy or False for train accuracy
    directory: str
        location of the data files and figures
    """

    # Load accuracy data
    data = load_all_accuracy(
        directory=directory,
        dataset_name=dataset_name,
        model_name=model_name,
        criterion_name=criterion_name,
        gamma1_list=gamma1_list,
        gamma2_list=gamma2_list,
        gamma3_list=None,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        hidden_units_3=None,
        epochs=epochs,
        batch_size=batch_size,
        is_test_accuracy=is_test_accuracy)

    for gamma_1 in gamma1_list:
        
        # Create figure and plot data
        fig = plt.figure(figsize=(20, 10))
        ax = data[gamma_1].plot()
        ax.set_ylim([0,1])
    
        # Set title, label legend and x- and y-axes
        ax.set_title('gamma_1={0}'.format(gamma_1))
        plt.legend(title='gamma_2', loc='center left', bbox_to_anchor=(1,0.5))
        plt.xlabel('Number of Epochs')
        y_label = 'Test Accuracy' if is_test_accuracy else 'Train Accuracy'
        plt.ylabel(y_label)
    
        
        # Generate file name
        train_test = 'test' if is_test_accuracy else 'train'
        fname = 'plot_{dataset_name}_{model_name}_gI{gamma_1}_hI{hidden_1}_hII{hidden_2}_e{epochs}_b{batch_size}_{train_test}'.format(
            dataset_name=dataset_name,
            model_name=model_name,
            gamma_1=gamma_1,
            hidden_1=hidden_units_1,
            hidden_2=hidden_units_2,
            epochs=epochs,
            batch_size=batch_size,
            train_test=train_test)
    
        # Generate full path
        figures_directory = os.path.join(directory, 'figures/')
        validate_directory(directory=figures_directory)
        path = os.path.join(figures_directory, fname)
        
        # Save figure
        ax.figure.savefig(path + '.png', dpi=300, bbox_inches='tight')
        ax.figure.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
        msg = "Successfully saved to {fname}".format(fname=fname)
        print(msg)        
        
    for gamma_2 in gamma2_list:
        
        # Create figure and plot data
        fig = plt.figure(figsize=(20, 10))
        ax = data.xs(gamma_2, axis=1, level=1).plot()
        ax.set_ylim([0,1])
    
        # Set title, label legend and x- and y-axes
        ax.set_title('gamma_2={0}'.format(gamma_2))
        plt.legend(title='gamma_1', loc='center left', bbox_to_anchor=(1,0.5))
        plt.xlabel('Number of Epochs')
        y_label = 'Test Accuracy' if is_test_accuracy else 'Train Accuracy'
        plt.ylabel(y_label)
        
        # Generate file name
        train_test = 'test' if is_test_accuracy else 'train'
        fname = 'plot_{dataset_name}_{model_name}_gII{gamma_2}_hI{hidden_1}_hII{hidden_2}_e{epochs}_b{batch_size}_{train_test}'.format(
            dataset_name=dataset_name,
            model_name=model_name,
            gamma_2=gamma_2,
            hidden_1=hidden_units_1,
            hidden_2=hidden_units_2,
            epochs=epochs,
            batch_size=batch_size,
            train_test=train_test)
    
        # Generate full path
        figures_directory = os.path.join(directory, 'figures/')
        validate_directory(directory=figures_directory)
        path = os.path.join(figures_directory, fname)
        
        # Save figure
        ax.figure.savefig(path + '.png', dpi=300, bbox_inches='tight')
        ax.figure.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
        msg = "Successfully saved to {fname}".format(fname=fname)
        print(msg)
    return


def run_2layer_accuracy_plots_multiple_hidden_units(dataset_name, model_name,
                                             criterion_name, gamma_1, gamma_2,
                                             hidden_units_list_1, hidden_units_list_2,
                                             epochs, batch_size, is_test_accuracy, directory):
    """
    Plots and saves figures of test or train accuracy which compare 
    different pairs of hidden units for fixed gammas for MLP2.    
    
    Parameters
    ----------
    dataset_name: str
        'mnist' 
    model_name: str
        'mlp2' 
    criterion_name: str
        'ce' (for Cross Entropy loss) or 'mse' (for Mean Squared Error loss)
    gamma_1: float
        the mean-field scaling parameter for the first layer
    gamma_2: float
        the mean-field scaling parameter for the second layer  
    hidden_units_list_1: ints
        a list of hidden units for the first layer
    hidden_units_2: int
        a list of hidden units for the second layer
    epochs: int
        number of times to iterate through the data set for training the model 
        and calculating accuracy
    batch_size: int
        the number of images per batch
    is_test_accuracy: bool
        True for test accuracy or False for train accuracy
    directory: str
        location of the data files and figures
    """
    
    column = 'Test' if is_test_accuracy else 'Train'
    
    dict_data = dict()
    
    for hidden_units_1, hidden_units_2 in zip(hidden_units_list_1, hidden_units_list_2):
    
        dict_data[(hidden_units_1, hidden_units_2)] = load_accuracy_for_single_gamma(
            directory=directory, 
            dataset_name=dataset_name, 
            model_name=model_name, 
            criterion_name=criterion_name, 
            gamma_1=gamma_1, 
            gamma_2=gamma_2, 
            gamma_3=None,
            hidden_units_1=hidden_units_1, 
            hidden_units_2=hidden_units_2, 
            hidden_units_3=None,
            epochs=epochs, 
            batch_size=batch_size)[column]
                
    data = pd.concat({
        'N1={h1},N2={h2}'.format(h1=hidden_units[0], h2=hidden_units[1]): dict_data[hidden_units]
        for hidden_units in dict_data.keys()
        }, axis=1)
    
    # Create a new figure and plot accuracy data
    fig = plt.figure(figsize=(20, 10))
    ax = data.plot()
    ax.set_ylim([0,1])

    # Set title, label legend and x- and y-axes
    ax.set_title('{col} Accuracy:  gamma_1={g1}, gamma_2={g2}'.format(
        col=column, g1=gamma_1, g2=gamma_2))
    plt.legend(title='hidden units', loc='center left', bbox_to_anchor=(1,0.5))
    plt.xlabel('Number of Epochs')
    y_label = 'Test Accuracy' if is_test_accuracy else 'Train Accuracy'
    plt.ylabel(y_label)
   
    # Generate file name
    fname = 'plot_{dataset_name}_{model_name}_gI{gamma_1}_gII{gamma_2}_e{epochs}_b{batch_size}_{train_test}'.format(
        dataset_name=dataset_name,
        model_name=model_name,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        epochs=epochs,
        batch_size=batch_size,
        train_test=column.lower())
    
    # Generate full path
    figures_directory = os.path.join(directory, 'figures/')
    validate_directory(directory=figures_directory)
    path = os.path.join(figures_directory, fname)
    
    # Save figure
    ax.figure.savefig(path + '.png', dpi=300, bbox_inches='tight')
    ax.figure.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
    msg = "Successfully saved to {fname}".format(fname=fname)
    print(msg)

    return



if __name__ == "__main__":    
    
    # PARAMETERS TO RUN LOCALLY
    dataset_name = 'mnist'
    model_name = 'mlp3'
    criterion_name = 'ce'
    gamma1_list = [0.5, 0.8]
    gamma2_list = [0.5, 0.8]
    gamma3_list =[0.5, 0.8]
    hidden_units_1 = 50
    hidden_units_2 = 50
    hidden_units_3 = 50
    epochs = 5
    batch_size = 20
    
    is_test_accuracy = True
    directory = '/usr2/postdoc/jyu32/Documents/9_16_2022'
        
    # Run run_3layer_accuracy_plots(...) for three-layer plots
    # Run run_2layer_accuracy_plots(...) for two-layer plots
    # Run run_2layer_accuracy_plots_multiple_hidden_units(...) to compare different hidden units combination for two-layer neural network
    run_3layer_accuracy_plots(
        dataset_name=dataset_name,
        model_name=model_name,
        criterion_name=criterion_name,
        gamma1_list = gamma1_list,
        gamma2_list = gamma2_list,
        gamma3_list = gamma3_list,
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        hidden_units_3=hidden_units_3,
        epochs=epochs,
        batch_size=batch_size,
        is_test_accuracy=is_test_accuracy,
        directory=directory)

