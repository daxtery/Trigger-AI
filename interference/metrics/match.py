import numpy
from scipy.spatial.distance import cosine


def similarity_metric(embedding1: numpy.ndarray, embedding2: numpy.ndarray) -> float:
    return numpy.nan_to_num(1 - cosine(embedding1, embedding2), 0)