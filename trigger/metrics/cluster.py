
from trigger.metrics.match import similarity_metric
from typing import TYPE_CHECKING

import numpy
if TYPE_CHECKING:
    from trigger.trigger_interface import TriggerInterface

def compute_cluster_score(interface: "TriggerInterface") -> float:
    node_scores = []

    for cluster_id in interface.processor.get_cluster_ids():

        tags_in_cluster = interface.processor.get_tags_in_cluster(cluster_id)
        number_of_tags = len(tags_in_cluster)

        if number_of_tags == 0:
            continue

        elif number_of_tags == 1:
            node_similarities = [1.0]

        else:
            instances_in_cluster = interface.get_instances_by_tag(tags_in_cluster) 

            node_similarities = [
                similarity_metric(test_instance.embedding, compare_instance.embedding)
                for i, test_instance in enumerate(instances_in_cluster[:-1])
                for compare_instance in instances_in_cluster[i + 1:]
            ]

        sim_mean = numpy.mean(node_similarities)
        sim_std = numpy.std(node_similarities)

        node_dispersion = sim_std / sim_mean

        node_dispersion_delta = (node_dispersion - 1) / (numpy.power(5, 0.5) / 5)

        node_delta = numpy.power(node_dispersion_delta, 2)

        node_score = numpy.exp(-(node_delta)) * numpy.log(number_of_tags)

        node_scores.append(node_score)

    return numpy.sum(node_scores)