import os
import torch
import matplotlib.pyplot as plt

def get_env_variable(var_name):
    """
    Retrieves an environment variable and raises an exception if it is not set.

    :param var_name: Name of the environment variable
    :return: The value of the environment variable
    """
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(f"Environment variable '{var_name}' not set.")
    return value


def plot_tensor(tensor, figsize=(10, 10)):
    """
    Plots a PyTorch tensor as an image with an optional figure size.
    
    Parameters:
    - tensor: A PyTorch tensor of shape (C, H, W) where C is the number of channels (3 for RGB),
      and H, W are the height and width of the image, respectively.
    - figsize: A tuple representing the figure size in inches, default is (10, 10).
    """
    if tensor.shape[0] == 3:  # Check if the tensor has three channels
        # Set the figure size
        plt.figure(figsize=figsize)
        
        # Convert tensor to numpy array and transpose from (C, H, W) to (H, W, C)
        image = tensor.numpy().transpose(1, 2, 0)
        
        # Plot the image
        plt.imshow(image)
        plt.axis('off')  # Hide the axes
        plt.show()
    else:
        print("The tensor does not have 3 channels.")
