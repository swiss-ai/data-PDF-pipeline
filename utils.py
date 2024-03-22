import os

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

