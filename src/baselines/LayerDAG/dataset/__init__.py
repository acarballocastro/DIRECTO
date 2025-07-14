from .layer_dag import *
from .general import DAGDataset
from .data import get_tpu_tile, get_synthetic


def load_dataset(dataset_name):
    if dataset_name == "tpu_tile":
        return get_tpu_tile()
    elif dataset_name in ["ba_dag", "er_dag", "er", "sbm", "planar"]:
        return get_synthetic(dataset_name)
    else:
        return NotImplementedError
