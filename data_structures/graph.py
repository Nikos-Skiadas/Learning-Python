import functools
import typing


Vertex = typing.TypeVar("Vertex", bound = typing.Hashable)
Weight = typing.TypeVar("Weight", bound = typing.Any)


@functools.total_ordering
class Node(
	dict[
		Vertex,
		Weight,
	]
):

	def __le__(self, node: typing.Self) -> bool:
		return self.keys() <= node.keys() and all(node[vertex] == weight for vertex, weight in self.items())


	@property
	def degree(self) -> int:
		return len(self)


Edge: typing.TypeAlias = tuple[
	Vertex,
	Vertex,
	Weight,
]


@functools.total_ordering
class Graph(
	Node[
		Vertex,
		Node[
			Vertex,
			Weight,
		],
	]
):

	def __le__(self, graph: typing.Self) -> bool:
		return self.keys() <= graph.keys() and all(self[vertex] <= graph[vertex] for vertex in self)


	@property
	def order(self) -> int:
		return len(self)

	@property
	def size(self) -> int:
		return sum(bool(self[tail][head]) for tail in self for head in self[tail])


	def adjacent(self,
		tail: Vertex,
		head: Vertex,
	) -> bool:
		return tail in self[head]

	def neighbors(self,
		tail: Vertex,
	) -> Node:
		return self[tail]

	def addNode(self,
		tail: Vertex,
	) -> None:
		self.setdefault(tail, Node())

	def addEdge(self,
		tail: Vertex,
		head: Vertex,
		edge: Weight,
	) -> None:
		self[tail][head] = edge

	def delNode(self,
		tail: Vertex,
	) -> None:
		del self[tail]

	def delEdge(self,
		tail: Vertex,
		head: Vertex,
	) -> None:
		del self[tail][head]


class BiGraph(
	Graph[
		Vertex,
		Weight,
	]
):

	def addEdge(self,
		tail: Vertex,
		head: Vertex,
		edge: Weight,
	) -> None:
		super().addEdge(
			tail,
			head,
			edge,
		)
		super().addEdge(
			head,
			tail,
			edge,
		)

	def delNode(self,
		tail: Vertex,
	) -> None:
		for head in self[tail]:
			super().delEdge(
				head,
				tail,
			)

		super().delNode(tail)

	def delEdge(self,
		tail: Vertex,
		head: Vertex,
	) -> None:
		super().delEdge(
			tail,
			head,
		)
		super().delEdge(
			head,
			tail,
		)
