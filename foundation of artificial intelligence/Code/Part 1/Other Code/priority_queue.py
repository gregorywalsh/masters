from heapq import heappush, heappop
from itertools import count

class PriorityQueue:

    REMOVED = '<removed-task>'

    def __init__(self):
        self.elements = []
        self.entry_finder = {}
        self.counter = count()

    def add_task(self, task, priority=0):
        """Add a new task or update the priority of an existing task"""
        if task in self.entry_finder:
            self.remove_task(task)

        tiebreaker = next(self.counter)
        entry = [priority, tiebreaker, task]
        self.entry_finder[task] = entry
        heappush(self.elements, entry)

    def remove_task(self, task):
        """Mark an existing task as REMOVED.  Raise KeyError if not found."""
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_task(self):
        """Remove and return the lowest priority task. Raise KeyError if empty."""
        while self.elements:
            priority, tiebreaker, task = heappop(self.elements)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task

        raise KeyError('pop from an empty priority queue')

    def get_task_priority(self, task):
        """Returns the priority of a given task"""
        return self.entry_finder[task][0]

    def is_empty(self):
        return len(self.elements) == 0

    def __iter__(self):
        for key in self.entry_finder:
            yield key