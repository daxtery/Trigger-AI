from trigger.train.cluster.gturbo.node import Node

class Link:

    def __init__(self, v: Node, u: Node) -> None:

        self.age = 0
        self.v = v
        self.u = u

    def fade(self) -> None:

        self.age += 1

    def renew(self) -> None:

        self.age = 0