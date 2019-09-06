from random import choice as rand_choice
from collections import deque

def negamax(state, remaining_plys, player):

    negamax.counter += 1

    # BASE CASES
    # Is a leaf node
    end_state = state.end_state()
    if end_state:
        if end_state == 'lose':
            return -float('inf'), None
        else:
            return 0, None

    # Is max depth
    if remaining_plys == 0:
        return player * state.evaluate(), None

    # RECURSIVE CASE
    best_score_yet = -float('inf')
    best_actions_yet = []

    for possible_action in state.possible_actions():

        child_state = state.execute_action(possible_action)

        child_state_score = -negamax(child_state, remaining_plys - 1, -player)[0]

        negamax.edges.setdefault(state.id, []).append( (child_state.id, possible_action, child_state_score, player, None, None) )

        if child_state_score > best_score_yet:
                best_score_yet = child_state_score
                best_actions_yet = [possible_action]
        elif child_state_score == best_score_yet:
                best_actions_yet.append(possible_action)

    return best_score_yet, rand_choice(best_actions_yet)


def negamax_ab(state, remaining_plys, player, alpha=-float('inf'), beta=float('inf')):

    negamax_ab.counter += 1

    # BASE CASES
    # Is a leaf node
    end_state = state.end_state()
    if end_state:
        if end_state == 'lose':
            return -float('inf'), None
        else:
            return 0, None

    # Is max depth
    if remaining_plys == 0:
        return player * state.evaluate(), None

    # RECURSIVE CASE
    best_score_yet = -float('inf')
    best_actions_yet = []

    for possible_action in state.possible_actions():

        child_state = state.execute_action(possible_action)

        child_state_score = -negamax_ab(child_state, remaining_plys - 1, -player, -beta, -alpha)[0]

        if child_state_score > best_score_yet:
                best_score_yet = child_state_score
                best_actions_yet = [possible_action]
        elif child_state_score == best_score_yet:
                best_actions_yet.append(possible_action)

        alpha = max(alpha, best_score_yet)

        negamax_ab.edges.setdefault(state.id, []).append( (child_state.id, possible_action, child_state_score, player, alpha, beta) )

        if alpha >= beta:
            break

    return best_score_yet, rand_choice(best_actions_yet)


def negamax_ab_with_ordering(state, remaining_plys, player, alpha=-float('inf'), beta=float('inf')):

    negamax_ab_with_ordering.counter += 1

    # BASE CASES
    # Is a leaf node
    end_state = state.end_state()
    if end_state:
        if end_state == 'lose':
            return -float('inf'), None
        else:
            return 0, None

    # Is max depth
    if remaining_plys == 0:
        return player * state.evaluate(), None

    # RECURSIVE CASE
    best_score_yet = -float('inf')
    best_actions_yet = []

    for possible_action in state.sorted_possible_actions():

        child_state = state.execute_action(possible_action)

        child_state_score = -negamax_ab_with_ordering(child_state, remaining_plys - 1, -player, -beta, -alpha)[0]

        if child_state_score > best_score_yet:
                best_score_yet = child_state_score
                best_actions_yet = [possible_action]
        elif child_state_score == best_score_yet:
                best_actions_yet.append(possible_action)

        alpha = max(alpha, best_score_yet)

        negamax_ab_with_ordering.edges.setdefault(state.id, []).append( (child_state.id, possible_action, child_state_score, player, alpha, beta) )

        if alpha >= beta:
            break

    return best_score_yet, rand_choice(best_actions_yet)


def print_graph(edge_dict, score, action, player):

    def get_player_name():
        return 1 if player == 1 else 2

    depth = 0
    first_node_id = 0
    queue = deque([(first_node_id, depth, score, action, player, None, None)])
    rows_printed = 1
    print('Turn: Player %d' % (get_player_name(),))
    print('Chosen action: column', action + 1)
    print('Score:', score, '\n')
    print('==== Action Tree ====', '\n')

    while queue:

        node_id, depth, score, action, player, alpha, beta = queue.pop()
        player_name = get_player_name()
        if depth > 0:
            if alpha:
                print("Node %d%sPlayer %d  |  Action: column %d  |  Score: %f  |  Alpha: %f  |  Beta: %f" %
                      (rows_printed, '\t\t' * (depth - 1), player_name, action + 1, score, alpha, beta))
            else:
                print("Node %d%sPlayer %d  |  Action: column %d  |  Score: %f" %
                      (rows_printed, '\t\t' * (depth - 1), player_name, action + 1, score))
            rows_printed += 1

        child_depth = depth + 1
        children = edge_dict.get(node_id, [])
        children.reverse()
        for child_id, action, child_score, player, alpha, beta in children:
            queue.append((child_id, child_depth, child_score, action, player, alpha, beta))



negamax.counter = 0
negamax.edges = {}

negamax_ab.counter = 0
negamax_ab.edges = {}

negamax_ab_with_ordering.counter = 0
negamax_ab_with_ordering.edges = {}