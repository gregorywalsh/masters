from collections import namedtuple
from heuristics import manhattan_dist


Position = namedtuple('Position', ['x', 'y'])


class Problem(object):

    def __init__(self, goal_test_function, initial_state, environment, agent, all_skills):
        self.goal_test_function = goal_test_function
        self.initial_state = initial_state
        self.environment = environment
        self.agent = agent
        self.all_skills = all_skills

    def permissible_skills(self, state=None):
        return self.all_skills

    def get_applicable_actions(self, state):
        actions = []

        for applicable_target, cost in self.agent.skill.get_targets(state=state, grid=self.environment):
            actions.append(self.agent.skill(applicable_target, cost))

        return actions

    def search(self):
        return self.agent.search_algorithm(self)


class Result(object):
    def __init__(self, state, cost):
        self.state = state
        self.cost = cost


class SwitchTilesAction(object):

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    @staticmethod
    def get_targets(state, grid):

        current_position = state.environment_data['agent_location']

        for x_move, y_move in SwitchTilesAction.moves:
            new_position = Position(current_position.x + x_move, current_position.y + y_move)
            if SwitchTilesAction.is_acceptable_position(new_position, grid):
                cost = manhattan_dist(current_position, new_position)
                yield new_position, cost


    @staticmethod
    def is_acceptable_position(position, grid):

        x_within_bounds = 0 <= position.x <= grid.width - 1
        y_within_bounds = 0 <= position.y <= grid.height - 1

        if x_within_bounds and y_within_bounds:
            return grid.is_passable(position)
        return False

    def __init__(self, agent_destination, cost=None):
        self.cost = cost
        self.agent_destination = agent_destination


    def execute(self, state):
        new_state = self.generate_new_state(state)
        return Result(state=new_state, cost=self.cost)


    def generate_new_state(self, state):


        # Record the location of the agent before the action
        new_state = State(environment_data={'agent_location': self.agent_destination, 'tiles': {}, 'goals': state.environment_data['goals']})

        # Update the positions of the tiles if they moved
        agent_location_before_action = state.environment_data['agent_location']
        for tile_key, position in state.environment_data['tiles'].items():
            if position == self.agent_destination:
                new_state.environment_data['tiles'][tile_key] = agent_location_before_action
            else:
                new_state.environment_data['tiles'][tile_key] = position

        return new_state


class State(object):

    def __init__(self, environment_data):
        self.environment_data = environment_data

    def __eq__(self, other):
        return type(self) is type(other) and self.environment_data == other.environment_data

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(frozenset(self.environment_data))


class Agent(object):

    def __init__(self, name, skill, search_algorithm):
        self.name = name
        self.skill = skill
        self.search_algorithm = search_algorithm


class Grid(object):

    def __init__(self, grid_data):
        self.grid_data = tuple(tuple(row) for row in grid_data)
        self.width = len(self.grid_data[0])
        self.height = len(self.grid_data)

    def is_passable(self, position):
        return self.grid_data[position.y][position.x] != 0
