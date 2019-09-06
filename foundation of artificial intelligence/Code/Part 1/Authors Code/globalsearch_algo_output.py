from collections import deque
from random import shuffle
from itertools import count
from heapq import heappop, heappush


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


def extract_action_sequence(action_edges, terminating_action):

    action_sequence = [terminating_action]
    parent = action_edges.get(terminating_action, None)

    while parent:
        action_sequence.append(parent)
        parent = action_edges.get(parent, None)

    action_sequence.reverse()

    return action_sequence


def depth_first_tree_search(problem, max_depth=float('inf')):

    # Initiate return variables
    action_sequence = []
    num_nodes_expanded = 0
    current_ply_depth = 0
    state_incrementer = 0

    # First check if start state is the goal state
    if problem.goal_test_function(problem.initial_state):
        return problem.initial_state, num_nodes_expanded, current_ply_depth, action_sequence
    else:

        # Initiate queue
        parent_action = None
        queue = deque([(problem.initial_state, state_incrementer, current_ply_depth, parent_action)])

        while queue:

            # Track the number of nodes expanded
            if num_nodes_expanded % 10000 == 0:
                pass #print('Number of nodes expanded: ', num_nodes_expanded)
            num_nodes_expanded += 1

            # Get the next node to expand
            state, state_id, current_ply_depth, parent_action = queue.pop()
            child_ply_depth = current_ply_depth + 1

            # Backtrack/update the action sequence
            num_prior_actions = current_ply_depth - 1
            prior_actions = action_sequence[:num_prior_actions]
            action_sequence = prior_actions + [parent_action]

            # Determine if max_depth will be exceeded when running in iterative deepening mode
            if child_ply_depth <= max_depth:

                # Determine applicable actions
                applicable_actions = problem.get_applicable_actions(state)
                shuffle(applicable_actions)

                if num_nodes_expanded <= 5:
                    ith = ['First', 'Second', 'Third', 'Fourth', 'Fifth'][num_nodes_expanded-1]
                    print("="*45, ith, "node expanded", "="*45, '\n')
                    print("Expanded node's state: ", end='')
                    print("ID", state_id, '  ||  Agent at', state.environment_data['agent_location'], end="")
                    for tile_key, position in state.environment_data['tiles'].items():
                        print('  ||  ', end='')
                        print(tile_key, "at", position, end='')
                    children_generated = 0

                # Check all child nodes for goal state, and add failures to the queue
                for child_action in applicable_actions:
                    state_incrementer +=1
                    child_state_id = state_incrementer
                    child_result = child_action.execute(state)
                    child_state = child_result.state

                    # Test for the goal state
                    if problem.goal_test_function(child_state):
                        # If goal state then return sequence
                        return child_state, num_nodes_expanded, current_ply_depth, action_sequence[1:] + [child_action]

                    else:
                        queue.append((child_state, child_state_id, child_ply_depth, child_action))
                        if num_nodes_expanded <= 5:
                            children_generated +=1
                            ith = ['First', 'Second', 'Third', 'Fourth', 'Fifth'][children_generated-1]
                            print('\n')
                            print("=" * 20, ith, "child generated", "=" * 20)
                            print('Agent move to:', child_action.agent_destination)
                            print("New state: ", end='')
                            print("ID", child_state_id, '  ||  Agent at', child_state.environment_data['agent_location'], end="")
                            for tile_key, position in child_state.environment_data['tiles'].items():
                                print('  ||  ', end='')
                                print(tile_key, "at", position, end='')

                            print('\n')
                            print('Updated Queue', end='')
                            queue_index = 0
                            for node in queue:
                                queue_index += 1
                                print('\nPosition', queue_index, 'state: ', end='')
                                print("ID", node[1], '  ||  Agent at', node[0].environment_data['agent_location'], end="")
                                for tile_key, position in node[0].environment_data['tiles'].items():
                                    print('  ||  ', end='')
                                    print(tile_key, "at", position, end='')

                if num_nodes_expanded <= 5: print('\n\n')

        # If no solution is found return None
        return None, num_nodes_expanded


def iterative_deepening_tree_search(problem, max_depth=float('inf')):

    # Initiate loop variables
    iteration_max_depth = 1
    total_num_nodes_expanded = 0
    nodes_per_level = []
    while iteration_max_depth <= max_depth:

        if iteration_max_depth <=4:
            print('#' * 100)
            print("Searching to a depth of",iteration_max_depth)
            print('#' * 100)
            print('\n')

        # Get depth limited result
        result = depth_first_tree_search(problem, max_depth=iteration_max_depth)
        total_num_nodes_expanded += result[1]
        nodes_per_level.append(result[1])

        # Return the result if not None, otherwise continue searching deeper
        if result[0]:
            goal_state, num_nodes_expanded, current_ply_depth, action_sequence = result
            return goal_state, total_num_nodes_expanded, current_ply_depth, action_sequence
        else:
            iteration_max_depth += 1

    # If no solution is found at max_depth return None
    return None, total_num_nodes_expanded


def breadth_first_tree_search(problem):

    # Initiate return variables
    action_sequence = []
    num_nodes_expanded = 0
    current_ply_depth = 0
    parent_action = None
    state_incrementer = 0

    # First check if start state is the goal state
    if problem.goal_test_function(problem.initial_state):
        return problem.initial_state, num_nodes_expanded, current_ply_depth, action_sequence
    else:

        # Initiate queue and tree data structure
        queue = deque([(problem.initial_state, state_incrementer, current_ply_depth, parent_action)])
        action_tree_edges = {}

        while queue:

            # Track the number of nodes expanded
            if num_nodes_expanded % 10000 == 0:
                print('Number of nodes expanded: ', num_nodes_expanded)
            num_nodes_expanded += 1

            # Get the next node to expand
            state, state_id, current_ply_depth, parent_action = queue.pop()
            child_ply_depth = current_ply_depth + 1

            if num_nodes_expanded <= 5:
                ith = ['First', 'Second', 'Third', 'Fourth', 'Fifth'][num_nodes_expanded - 1]
                print("=" * 45, ith, "node expanded", "=" * 45, '\n')
                print("Expanded node's state: ", end='')
                print("ID", state_id, '  ||  Agent at', state.environment_data['agent_location'], end="")
                for tile_key, position in state.environment_data['tiles'].items():
                    print('  ||  ', end='')
                    print(tile_key, "at", position, end='')
                children_generated = 0

            # Expand the current nodes and add children to the queue
            for child_action in problem.get_applicable_actions(state):

                state_incrementer +=1
                child_state_id = state_incrementer

                # Store the action edge
                action_tree_edges[child_action] = parent_action

                # Generate the child state
                child_result = child_action.execute(state)
                child_state = child_result.state

                # Test for the goal state
                if problem.goal_test_function(child_state):
                    # Calculate the action sequence
                    action_sequence = extract_action_sequence(action_edges=action_tree_edges, terminating_action=child_action)
                    return child_state, num_nodes_expanded, current_ply_depth, action_sequence
                else:
                    queue.appendleft((child_state, child_state_id, child_ply_depth, child_action))
                    if num_nodes_expanded <= 5:
                        children_generated +=1
                        ith = ['First', 'Second', 'Third', 'Fourth', 'Fifth'][children_generated-1]
                        print('\n')
                        print("=" * 20, ith, "child generated", "=" * 20)
                        print('Agent move to:', child_action.agent_destination)
                        print("New state: ", end='')
                        print("ID", child_state_id, '  ||  Agent at', child_state.environment_data['agent_location'], end="")
                        for tile_key, position in child_state.environment_data['tiles'].items():
                            print('  ||  ', end='')
                            print(tile_key, "at", position, end='')

                        print('\n')
                        print('Updated Queue', end='')
                        queue_index = 0
                        for node in queue:
                            queue_index += 1
                            print('\nPosition', queue_index, 'state: ', end='')
                            print("ID", node[1], '  ||  Agent at', node[0].environment_data['agent_location'], end="")
                            for tile_key, position in node[0].environment_data['tiles'].items():
                                print('  ||  ', end='')
                                print(tile_key, "at", position, end='')

            if num_nodes_expanded <= 5: print('\n\n')

        # If no solution is found return None
        return None, num_nodes_expanded


def a_star_tree_search(problem, heuristic_function):

    # TODO TESTING

    # Initiate return variables
    num_nodes_expanded = 0
    current_ply_depth = 0
    action_tree_edges = {}
    state_incrementer = 0

    # Initiate loop variables
    estimated_total_cost = 0  # Value is irrelevant for root node since always popped first
    cost_to_current_ply = 0
    parent_action = None
    queue = PriorityQueue()
    queue.add_task((problem.initial_state, state_incrementer, current_ply_depth, cost_to_current_ply, parent_action), estimated_total_cost)

    while not queue.is_empty():

        # Track the number of nodes expanded
        if num_nodes_expanded % 10000 == 0:
            print('Number of nodes expanded: ', num_nodes_expanded)
        num_nodes_expanded += 1

        # Get the next node to explore
        state, state_id, current_ply_depth, cost_to_current_ply, parent_action = queue.pop_task()

        # Test for the goal state, with admissible heuristic this is guaranteed to be the optimal solution
        if problem.goal_test_function(state):
            action_sequence = extract_action_sequence(action_edges=action_tree_edges, terminating_action=parent_action)
            return state, num_nodes_expanded, current_ply_depth, action_sequence
        else:

            if num_nodes_expanded <= 5:
                ith = ['First', 'Second', 'Third', 'Fourth', 'Fifth'][num_nodes_expanded - 1]
                print("=" * 45, ith, "node expanded", "=" * 45, '\n')
                print("Expanded node's state: ", end='')
                print("ID", state_id, '  ||  Agent at', state.environment_data['agent_location'], end="")
                for tile_key, position in state.environment_data['tiles'].items():
                    print('  ||  ', end='')
                    print(tile_key, "at", position, end='')
                children_generated = 0

            # Expand the current nodes and add children to the queue
            for child_action in problem.get_applicable_actions(state):

                state_incrementer +=1
                child_state_id = state_incrementer

                # Store the action edge
                action_tree_edges[child_action] = parent_action

                # Generate the child state
                result = child_action.execute(state)
                child_state = result.state

                # Calculate the cost to this point
                cost_to_child_ply = cost_to_current_ply + result.cost
                estimated_total_cost = cost_to_child_ply + heuristic_function(child_state)

                # Insert next node into the queue
                child_ply_depth = current_ply_depth + 1
                queue.add_task((child_state, child_state_id, child_ply_depth, cost_to_child_ply, child_action), estimated_total_cost)

                if num_nodes_expanded <= 5:
                    children_generated +=1
                    ith = ['First', 'Second', 'Third', 'Fourth', 'Fifth'][children_generated-1]
                    print('\n')
                    print("=" * 20, ith, "child generated", "=" * 20)
                    print('Agent move to:', child_action.agent_destination)
                    print("New state: ", end='')
                    print("ID", child_state_id, '  ||  Agent at', child_state.environment_data['agent_location'], end="")
                    for tile_key, position in child_state.environment_data['tiles'].items():
                        print('  ||  ', end='')
                        print(tile_key, "at", position, end='')

                    print('\n')
                    print('Updated Heap', end='')
                    queue_index = 0
                    for node in queue:
                        task_priority = queue.get_task_priority(node)
                        print('\nState: ', end='')
                        print("ID", node[1], '  ||  Estimated Cost', task_priority, '  ||  Agent at', node[0].environment_data['agent_location'], end="")
                        for tile_key, position in node[0].environment_data['tiles'].items():
                            print('  ||  ', end='')
                            print(tile_key, "at", position, end='')

        if num_nodes_expanded <= 5: print('\n\n')

    # If no solution is found return None
    return None, num_nodes_expanded


def a_star_search(problem, heuristic_function):
    frontier = PriorityQueue()
    frontier.add_task(problem.initial_state, 0)
    explored = set()
    min_costs_hitherto = {problem.initial_state: 0}
    empty_action_sequence = []
    cheapest_action_sequence = {problem.initial_state: empty_action_sequence}

    while not frontier.is_empty():

        state = frontier.pop_task()
        if problem.goal_test_function(state):
            return cheapest_action_sequence[state]

        explored.add(state)
        for applicable_action in problem.get_applicable_actions(state):
            result = applicable_action.execute(state)
            new_cost = min_costs_hitherto[state] + result.cost
            estimated_cost = new_cost + heuristic_function(result.state)
            if (result.state not in explored and result.state not in frontier) \
                    or (result.state in frontier and estimated_cost < frontier.get_task_priority(result.state)):
                min_costs_hitherto[result.state] = new_cost
                cheapest_action_sequence[result.state] = cheapest_action_sequence[state] + [applicable_action]
                frontier.add_task(result.state, estimated_cost)

    return None