import logging
from trigger.metrics.match import similarity_metric
import numpy

from sklearn.metrics import silhouette_score
from typing import Dict, Any

import statistics
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('cluster')
logger.setLevel(logging.INFO)

from typing import TYPE_CHECKING
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

        node_score = numpy.ex(-(node_delta)) * numpy.log(number_of_tags)

        node_scores.append(node_score)

    return numpy.sum(node_scores)

def eval_cluster(interface: "TriggerInterface") -> Dict[str, Any]:

    tags = interface.instances_map.keys()
    labels = []
    instances = []
    labels_set = set()

    for tag in tags:
        label = interface.processor.get_cluster_by_tag(tag)

        labels.append(label)

        labels_set.add(label)
    
        instances.append(interface.instances_map[tag])

    try:

        Ss = silhouette_score(instances, labels)
    except:

        Ss = -1.0

    num_instances_per_cluster = [
        len(interface.processor.get_tags_in_cluster(cluster_id))
        for cluster_id in interface.processor.get_cluster_ids()
    ]

    counter = Counter(num_instances_per_cluster)

    return {
        'ss': float(Ss),
        'cluster_score': compute_cluster_score(interface),
        '#clusters': len(labels_set),
        '#instances': len(tags),
        'distribution': {score: count for score, count in counter.most_common()},
        'avg/mean instances / cluster': statistics.mean(num_instances_per_cluster),
        'max instances of any cluster': counter.most_common()[0:1][0][0] if len(counter) > 0 else 0,
        'min instances of any cluster': counter.most_common()[-2:][0][0] if len(counter) > 1 else 0
    }