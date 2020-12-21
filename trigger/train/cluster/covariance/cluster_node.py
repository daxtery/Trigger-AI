import numpy as np

class ClusterNode:

    def __init__(self, id, instance, initial_std) -> None:

        self.id = id
        self.cov_matrix = np.eye(1024)
        self.std = initial_std
        self.mean = instance
        self.instances = [instance]

        i_observation = instance.reshape((1024, 1))

        self.observations = i_observation

    def add_instance(self, instance) -> None:

        self.instances.append(instance)

        self.observations = np.hstack([self.observations, instance.reshape((1024, 1))])

        self.cov_matrix = np.cov(self.observations)

        self.mean = np.mean(self.instances, axis=0)

        std_vector = np.std(self.instances, axis=0)

        self.std = np.linalg.norm(std_vector)