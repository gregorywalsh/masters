from collections import deque
from random import shuffle
from priority_queue import PriorityQueue


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
    action_sequence = deque()
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
                print('Number of nodes expanded: ', num_nodes_expanded)
            num_nodes_expanded += 1

            # Get the next node to expand
            state, state_id, current_ply_depth, parent_action = queue.pop()
            child_ply_depth = current_ply_depth + 1

            # Backtrack/update the action sequence
            num_prior_actions = current_ply_depth
            for i in range(len(action_sequence) - num_prior_actions):
                action_sequence.pop()
            action_sequence.append(parent_action)

            # Determine if max_depth will be exceeded when running in iterative deepening mode
            if child_ply_depth <= max_depth:

                # Determine applicable actions
                applicable_actions = problem.get_applicable_actions(state)
                shuffle(applicable_actions)

                # Check all child nodes for goal state, and add failures to the queue
                for child_action in applicable_actions:
                    state_incrementer +=1
                    child_state_id = state_incrementer
                    child_result = child_action.execute(state)
                    child_state = child_result.state

                    # Test for the goal state
                    if problem.goal_test_function(child_state):
                        # If goal state then return sequence
                        actions = [action for action in action_sequence] + [child_action]
                        return child_state, num_nodes_expanded, current_ply_depth, actions[1:]

                    else:
                        queue.append((child_state, child_state_id, child_ply_depth, child_action))

        # If no solution is found return None
        return None, num_nodes_expanded


def iterative_deepening_tree_search(problem, max_depth=float('inf')):

    # Initiate loop variables
    iteration_max_depth = 1
    total_num_nodes_expanded = 0
    nodes_per_level = []
    while iteration_max_depth <= max_depth:

        # Get depth limited result
        result = depth_first_tree_search(problem, max_depth=iteration_max_depth)
        total_num_nodes_expanded += result[1]
        nodes_per_level.append(result[1])

        # Return the result if not None, otherwise continue searching deeper
        if result[0]:
            print(nodes_per_level)
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

        # If no solution is found return None
        return None, num_nodes_expanded


def a_star_tree_search(problem, heuristic_function):

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

    # If no solution is found return None
    return None, num_nodes_expanded
