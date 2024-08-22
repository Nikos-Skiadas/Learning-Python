from data_structures import sequential
from data_structures import graph


def empty(lines: int):
	for _ in range(lines):
		print()


class TestLinked:

	items = [
		2,
		3,
		5,
		7,
	]

	def test_stack(self):
		stack = sequential.Stack()

		for item in self.items:
			stack.push(item)

		assert self.items == list(stack)

		for item in reversed(self.items):
			assert item == stack.pop()

		assert not list(stack)

	def test_queue(self):
		queue = sequential.Queue()

		for item in self.items:
			queue.enqueue(item)

		assert self.items == list(queue)

		for item in self.items:
			assert item == queue.dequeue()

		assert not list(queue)

	def test_deque(self):
		deque = sequential.Deque()

		for item in self.items:
			deque.prepend(item)
			deque.append(item)

		assert list(reversed(self.items)) + self.items == list(deque)

		for item in reversed(self.items):
			assert item == deque.pull()
			assert item == deque.pop()

		assert not list(deque)


class TestGraph:

	unweighted_edges = {
		(1, 2),
		(1, 3),
		(1, 4),
		(1, 5),
		(2, 3),
		(2, 4),
		(2, 5),
		(3, 4),
		(3, 5),
		(4, 5),
	}
	edges = {(*edge, sum(edge)) for edge in unweighted_edges}

	def test_graph(self):
		empty(2)
		print(graph.Graph.from_edges(*self.edges))

	def test_unweighted_graph(self):
		empty(2)
		print(graph.UnweightedGraph.from_edges(*self.unweighted_edges))

	def test_undirected_graph(self):
		empty(2)
		print(graph.UndirectedGraph.from_edges(*self.edges))

	def test_undirected_unweighted_graph(self):
		empty(2)
		print(graph.UndirectedUnweightedGraph.from_edges(*self.unweighted_edges))
