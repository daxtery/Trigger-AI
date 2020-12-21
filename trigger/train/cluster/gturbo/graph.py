import heapq

from trigger.train.cluster.gturbo.link import Link
from trigger.train.cluster.gturbo.node import Node

from typing import List, Optional, Tuple, Dict

class Graph:

    def __init__(self) -> None:

        self.nodes: Dict[int, Node] = {}
        self.links: Dict[Tuple[int, int], Link] = {}
        self.heap = []

    def update_heap(self) -> None:

        heapq.heapify(self.heap)

    def insert_node(self, node: Node) -> None:

        self.nodes[node.id] = node
        heapq.heappush(self.heap, node)

    def remove_node(self, node: Node) -> None:

        for u in node.topological_neighbors.values():

            self.remove_link(node, u)

        self.nodes.pop(node.id)

        self.heap.remove(node)
        heapq.heapify(self.heap)
        
    def get_node(self, id) -> Node:

        return self.nodes[id]

    def insert_link(self, v: Node, u: Node, link) -> None:

        self.links[(v.id, u.id)] = link

    def remove_link(self, v: Node, u: Node) -> None:

        if self.links.get((v.id, u.id), None) != None:

            self.links.pop((v.id, u.id))

        elif self.links.get((u.id, v.id), None) != None:
            self.links.pop((u.id, v.id))
            

    def has_link(self, v: Node, u: Node) -> bool:

        return (self.links.get((v.id, u.id), None) != None) or (self.links.get((u.id, v.id), None) != None)
            
    def get_link(self, v: Node, u: Node) -> Optional[Link]:

        link = self.links.get((v.id, u.id), None)

        if link != None:

            return link

        return self.links.get((u.id, v.id), None)

    def get_q_and_f(self) -> Tuple[Node, Node]:

        q = self.heap[0]
        f = sorted(q.topological_neighbors.values(), 
                                    key=lambda node: node.error, reverse=True)[0]

        return (q, f)