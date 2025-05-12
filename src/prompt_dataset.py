"""
Class of the prompt dataset. 
"""

from fastparquet import ParquetFile
import numpy as np
from torch.utils.data import Dataset
from urllib.request import urlretrieve
from typing import Any
from warnings import warn

class DiffusionDBPrompts(Dataset):

    def __init__(self, path: str = None):
        self.prompt_array = self.__get_prompts(path)

    @staticmethod
    def __get_prompts(path: str) -> np.ndarray:
        if path is None:
            # This downloads the training dataset
            warn('Missing DiffusionDB dataset file is being downloaded in the current working directory. Be aware that it takes around 190 MB of memory.')
            path = 'data.parquet'
            table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
            urlretrieve(table_url, path)
        
        # Be careful to change the name of the columns of other prompt datasets.
        return ParquetFile(path).to_pandas()['prompt'].unique()
    
    def truncate(self, max_prompts: int):
        self.prompt_array = np.random.choice(self.prompt_array, max_prompts, replace=False)
    
    def __len__(self) -> int:
        return len(self.prompt_array)
    
    def __getitem__(self, index: int) -> str:
        return self.prompt_array[index]