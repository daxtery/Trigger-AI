import logging
from trigger.util.statistics import stats_from_counter
from trigger.metrics.cluster import compute_cluster_score

from sklearn.metrics import silhouette_score
from typing import Dict, Any, TYPE_CHECKING

from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('cluster')
logger.setLevel(logging.INFO)

if TYPE_CHECKING:
    from trigger.trigger_interface import TriggerInterface


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

        Ss = silhouette_score(instances, labels) # type: ignore
    except:

        Ss = -1.0

    num_instances_per_cluster = list(filter(lambda n: n > 0, 
    (
        len(interface.processor.get_tags_in_cluster(cluster_id))
        for cluster_id in interface.processor.get_cluster_ids()
    )))

    counter = Counter(num_instances_per_cluster)

    (distribution, int_stats) = stats_from_counter(counter)

    assert int_stats is not None

    avg, max, min = int_stats

    return {
        'ss': float(Ss),
        'cluster_score': compute_cluster_score(interface),
        '#clusters': len(labels_set),
        '#instances': len(tags),
        'distribution instances per cluster': distribution,
        'average instances per cluster': avg,
        'max instances per cluster': max,
        'min instances per cluster': min
    }