def blockworld_manhattan_boost(state):

    def manhattan_distance_to_closest_misplaced_tile(agent_location, tiles, goals):
        # check for misplaced tile and find the distance to the closest one
        min_distance_yet = None
        for tile_key, tile_position in tiles.items():
            if tile_position != goals[tile_key]:
                if min_distance_yet is None:
                    min_distance_yet = manhattan_dist(agent_location, tile_position) - 1
                else:
                    min_distance_yet = min(min_distance_yet, manhattan_dist(agent_location, tile_position) - 1)

        return min_distance_yet

    # Manhattan distance for all tiles
    tiles = state.environment_data['tiles']
    goals = state.environment_data['goals']
    agent_location = state.environment_data['agent_location']

    manhattan_cost = blockworld_manhattan(state)

    # Compute agent cost to nearest misplaced tile
    agent_cost = manhattan_distance_to_closest_misplaced_tile(agent_location=agent_location, tiles=tiles, goals=goals)
    # Check if is None since there may be no misplaced tiles
    if agent_cost:
        return manhattan_cost + agent_cost

    return manhattan_cost


def blockworld_manhattan(state):
    total_distance = 0
    for tile_key, position in state.environment_data['tiles'].items():
        total_distance += manhattan_dist(position, state.environment_data['goals'][tile_key])
    return total_distance


def blockworld_misplaced_tile_count(state):
    count_misplaced = 0
    for tile_key, position in state.environment_data['tiles'].items():
        if position != state.environment_data['goals'][tile_key]:
            count_misplaced += 1
    return count_misplaced


def manhattan_dist(old_position, new_position):
    x_dist = abs(old_position.x - new_position.x)
    y_dist = abs(old_position.y - new_position.y)
    return x_dist + y_dist
