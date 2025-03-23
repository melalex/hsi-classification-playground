import tqdm.notebook

from src.util.notebook import is_notebook


def create_progress_bar():
    if is_notebook():
        return tqdm.notebook.tqdm
    else:
        return tqdm.tqdm
