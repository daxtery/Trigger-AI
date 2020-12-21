from typing import List, Any
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import numpy

def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class SentenceEmbedder:

    def __init__(self, modelname:str='roberta-large-nli-stsb-mean-tokens'):

        self.model = SentenceTransformer(modelname)

    def generateEmbeddings(self, sentence: str) -> numpy.ndarray:

        return self.model.encode(sentence)


    def generateEmbeddingsFromList(self, skill_list: List[Any]) -> numpy.ndarray:
        if len(skill_list) == 0:

            return numpy.zeros(1024)

        skill_names = numpy.array([
            skill.name
            for skill in skill_list
        ])

        skills_embedding = []

        for batch in chunks(skill_names, 32):
            skills_embedding.extend(self.model.encode(batch, show_progress_bar=False))

        if len(skills_embedding) == 1:

            return skills_embedding[0]

        else:

            return tf.keras.layers.Average()(skills_embedding).numpy()