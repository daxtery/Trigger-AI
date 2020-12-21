import logging

from trigger.train.cluster.Processor import Processor

from sklearn.metrics import silhouette_score
from typing import Dict, Any

import statistics
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('cluster')
logger.setLevel(logging.INFO)


def eval_cluster(cluster: Processor) -> Dict[str, Any]:
    # logger.info("Computing cluster results...")

    X, tags = cluster.get_all_instances_with_tags()
    labels = []
    labels_set = set()

    for tag in tags:
        label = cluster.get_cluster_by_tag(tag)
        labels.append(label)
        labels_set.add(label)

    try:

        Ss = silhouette_score(X, labels)
    except:

        Ss = -1.0

    num_instances_per_cluster = [
        len(cluster.get_instances_and_tags_in_cluster(label)[1]) for label in labels_set
    ]

    counter = Counter(num_instances_per_cluster)

    return {
        'ss': float(Ss),
        'cluster_score': cluster.compute_cluster_score(),
        '#clusters': len(labels_set),
        '#instances': len(tags),
        'distribution': {score: count for score, count in counter.most_common()},
        'avg/mean instances / cluster': statistics.mean(num_instances_per_cluster),
        'max instances of any cluster': counter.most_common()[0:1][0][0] if len(counter) > 0 else 0,
        'min instances of any cluster': counter.most_common()[-1:][0][0] if len(counter) > 0 else 0
    }