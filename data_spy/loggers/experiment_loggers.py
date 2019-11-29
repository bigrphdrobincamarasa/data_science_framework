"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

TODO: doc
**  **
"""
def experiment_launcher(
        project_root: str, output_root: str, module: list
):
    """
    Function that launches experiments

    :param project_root:
    :param output_root:
    :param module:
    :return:
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            # Call function
            f(*args, **kwargs)
        return wrapper
    return decorator


def experiment_launcher(
        project_root: str, output_root: str, module: list
):
    """
    Function that launches experiments
    TODO doc

    :param project_root:
    :param output_root:
    :param module:
    :return:
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            # Call function
            f(*args, **kwargs)
        return wrapper
    return decorator

