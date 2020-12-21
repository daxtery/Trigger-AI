import numpy
from trigger.train.transformers.transformer_pipeline import Instance, TransformerPipeline

class IdentityPipeline(TransformerPipeline[numpy.ndarray]):

    def transform(self, value: numpy.ndarray) -> Instance[numpy.ndarray]:
        assert isinstance(value, numpy.ndarray)
        return Instance(value, value)