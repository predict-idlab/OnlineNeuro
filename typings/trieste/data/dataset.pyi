from typing import Sized

class Dataset(Sized):
    def __len__(self) -> int: ...
