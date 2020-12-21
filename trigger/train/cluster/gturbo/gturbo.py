from typing import Any, List, Tuple, Optional, Dict

import numpy
import numpy as np
import faiss

from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist

from trigger.models.match import Match

from trigger.train.cluster.gturbo.graph import Graph
from trigger.train.cluster.gturbo.node import Node
from trigger.train.cluster.gturbo.link import Link
from trigger.train.transformers.opening_transformer import OpeningInstance

from trigger.train.cluster.Processor import Processor


class GTurbo(Processor):

    def __init__(self, epsilon_b: float, epsilon_n: float, lam: int, beta: float,
                 alpha: float, max_age: int, r0: float,
                 dimensions: int = 2, random_state: int = 42) -> None:

        self.graph = Graph()

        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.lam = lam
        self.beta = beta
        self.alpha = alpha
        self.max_age = max_age
        self.dimensions = dimensions
        self.r0 = r0

        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dimensions))

        self.next_id = 2
        self.point_to_cluster = {}
        self.instances = {}
        self.tag_to_data = {}

        self.cycle = 0
        self.step = 1

        np.random.seed(random_state)

        node_1 = Node(np.random.rand(1, dimensions).astype(
            'float32')[0], 0, id=0, error_cycle=0, radius=r0)
        node_2 = Node(np.random.rand(1, dimensions).astype(
            'float32')[0], 0, id=1, error_cycle=0, radius=r0)

        self.graph.insert_node(node_1)
        self.graph.insert_node(node_2)

        self.index.add_with_ids(
            np.array([node_1.protype, node_2.protype]), np.array([0, 1]))

    def turbo_step(self, tag, instance):

        self.turbo_adapt(tag, instance)

        if self.step == self.lam - 1:

            self.turbo_increase()
            self.cycle += 1
            self.step = 0

        else:

            self.step += 1

    def create_node(self, q: Node, f: Node, radius: float) -> Node:

        r = Node(np.array(0.5*(q.protype + f.protype)).astype(
            'float32'), 0,
            self.next_id, self.cycle, radius)
        self.next_id += 1

        self.graph.insert_node(r)
        self.index.add_with_ids(np.array([r.protype]), np.array([r.id]))

        return r

    def create_node_from_instance(self, instance, radius: float) -> Node:

        r = Node(instance.astype(
            'float32'), 0, self.next_id, self.cycle, radius)
        self.next_id += 1

        self.graph.insert_node(r)
        self.index.add_with_ids(np.array([r.protype]), np.array([r.id]))

        return r

    def turbo_increase(self) -> None:

        q, f = self.graph.get_q_and_f()

        r = self.create_node(q, f, self.r0)

        q.remove_neighbor(f)
        f.remove_neighbor(q)

        self.graph.remove_link(q, f)

        self.create_link(q, r)
        self.create_link(f, r)

        self.decrease_error(q)
        self.decrease_error(f)

        r.error = 0.5*(q.error + f.error)
        self.graph.update_heap()

    def get_best_match(self, instance) -> Tuple[Node, Node]:

        _, I = self.index.search(np.array([instance]).astype('float32'), 2)

        return (self.graph.get_node(I[0][0]), self.graph.get_node(I[0][1]))

    def increment_error(self, node: Node, value: float) -> None:

        self.fix_error(node)
        node.error = node.error * \
            (np.power(self.beta, self.lam - self.step)) + value

        self.graph.update_heap()

    def fix_error(self, node: Node) -> None:

        node.error = np.power(self.beta, self.lam *
                              (self.cycle - node.error_cycle))*node.error
        node.update_error_cycle(self.cycle)

    def update_prototype(self, v: Node, scale: float, instance) -> None:

        self.index.remove_ids(np.array([v.id]))

        v.protype += scale*(instance - v.protype)

        self.index.add_with_ids(np.array([v.protype]), np.array([v.id]))

    def distance(self, u, v) -> float:

        return cdist([u], [v], "euclidean")[0]

    def turbo_adapt(self, tag: str, instance):

        v, u = self.get_best_match(instance)

        if self.distance(v.protype, instance) <= v.radius:

            v.add_instance(tag)
            self.instances[tag] = instance

            self.point_to_cluster[tag] = v.id

            error_value = np.power(self.distance(
                v.protype, instance), 2)[0]

            self.increment_error(v, error_value)

            self.update_prototype(v, self.epsilon_b, instance)

            for node in v.topological_neighbors.values():

                self.update_prototype(node, self.epsilon_n, instance)

            self.age_links(v)

            if not self.graph.has_link(v, u):

                self.create_link(v, u)

            link = self.graph.get_link(v, u)
            link.renew()

            self.update_edges(v)
            self.update_nodes()

        else:

            r = self.create_node_from_instance(instance, self.r0)
            r.add_instance(tag)

            self.instances[tag] = instance
            self.point_to_cluster[tag] = r.id

            self.create_link(v, r)

    def decrease_error(self, v: Node) -> None:

        self.fix_error(v)

        v.error *= self.alpha
        self.graph.update_heap()

    def create_link(self, v: Node, u: Node) -> None:

        link = Link(v, u)

        self.graph.insert_link(v, u, link)

        v.add_neighbor(u)
        u.add_neighbor(v)

    def update_nodes(self) -> None:

        nodes_to_remove = []

        for node in self.graph.nodes.values():

            if len(node.topological_neighbors) == 0 and len(node.instances) == 0:

                nodes_to_remove.append(node)

        for node in nodes_to_remove:

            self.graph.remove_node(node)
            self.index.remove_ids(np.array([node.id]))

    def age_links(self, v: Node) -> None:

        for u in v.topological_neighbors.values():

            link = self.graph.get_link(v, u)

            link.fade()

    def update_edges(self, v: Node) -> None:

        links_to_remove = []

        for u in v.topological_neighbors.values():

            link = self.graph.get_link(v, u)

            if link.age > self.max_age:

                links_to_remove.append((v, u))

        for v, u in links_to_remove:

            v.remove_neighbor(u)
            u.remove_neighbor(v)

            self.graph.remove_link(v, u)

    def get_cluster(self, instance) -> int:

        return self.point_to_cluster.get(instance)

    def process(self, tag: str, instance: np.ndarray, custom_data: Any = None) -> None:

        self.turbo_step(tag, instance)
        self.tag_to_data[tag] = custom_data

    def update(self, tag: str, instance: np.ndarray, custom_data: Any = None) -> None:

        self.remove(tag)

        self.process(tag, instance, custom_data)

    def remove(self, tag: str) -> None:

        node_id = self.point_to_cluster.pop(tag)
        self.instances.pop(tag)
        self.tag_to_data.pop(tag)

        node = self.graph.get_node(node_id)
        node.remove_instance(tag)

    def get_cluster_by_tag(self, tag: str) -> Optional[int]:

        return self.point_to_cluster[tag]

    def get_custom_data_by_tag(self, tag: str) -> Optional[Any]:

        return self.tag_to_data[tag]

    def get_instance_by_tag(self, tag: str) -> Optional[numpy.ndarray]:

        return self.instances.get(tag, None)

    def get_instances_and_tags_in_cluster(self, cluster_id: int) -> Tuple[List[np.ndarray], List[str]]:

        tags = [tag for tag, id in self.point_to_cluster.items() if id ==
                cluster_id]
        instances = [self.instances[tag] for tag in tags]

        return (instances, tags)

    def get_all_instances_with_tags(self) -> Tuple[List[np.ndarray], List[str]]:

        instances = [instance for instance in self.instances.values()]
        tags = [tag for tag in self.instances.keys()]

        return (instances, tags)

    def predict(self, instance: np.ndarray) -> int:

        return self.get_best_match(instance)[0].id

    def describe(self) -> Dict[str, Any]:

        return {
            "name": "GTurbo",
            "parameters": {
                "epsilon_b": self.epsilon_b,
                "epsilon_n": self.epsilon_n,
                "lam": self.lam,
                "beta": self. beta,
                "alpha": self.alpha,
                "max_age": self.max_age,
                "radius": self.r0
            }
        }

    def safe_file_name(self) -> str:

        return f"GTurbo = epsilon_b={self.epsilon_b};epsilon_n={self.epsilon_n};lam={self.lam};beta={self.beta};alpha={self.alpha};max_age={self.max_age};radius={self.r0}"

    def compute_cluster_score(self) -> float:

        instance_dist = self._get_instances_per_node()

        node_scores = []

        for node in self.graph.nodes.values():

            if len(node.instances) > 0:

                node_dispersion_delta = self._compute_node_delta(node)
                node_delta = np.power(node_dispersion_delta, 2)

                node_score = np.exp(-(node_delta))*np.log(len(node.instances))
                node_scores.append(node_score)

        return np.sum(node_scores)

    def _get_instances_per_node(self):

        return [len(node.instances) for node in self.graph.nodes.values() if len(node.instances) > 0]

    def _mean_instances_per_node(self, num_instances):

        return np.mean(num_instances)

    def _std_instances_per_node(self, num_instances):
        
        return np.std(num_instances)

    def _compute_node_delta(self, node):

        instances = node.instances

        if len(node.instances) == 1:

            node_similarities = [1.0]

        else:

            node_similarities = self._get_similarities(instances)

        sim_mean = self._get_mean_similarity(node_similarities)
        sim_std = self._get_std_similarity(node_similarities)

        node_dispersion = sim_std / sim_mean

        return (node_dispersion - 1) / (np.power(5, 0.5) / 5)

    def _get_similarities(self, instances):

        node_similarities = []

        for i in range(len(instances) - 1):

            test_instance = self.instances[instances[i]]

            for j in range(i + 1, len(instances)):

                compare_instance = self.instances[instances[j]]

                cos_sim = 1 - cosine(test_instance, compare_instance)
                node_similarities.append(cos_sim)

        return node_similarities

    def _get_mean_similarity(self, similarities):

        return np.mean(similarities)

    def _get_std_similarity(self, similarities):

        return np.var(similarities)