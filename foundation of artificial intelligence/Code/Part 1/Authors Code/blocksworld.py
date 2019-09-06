from collections import namedtuple
from globalsearch import a_star_tree_search, depth_first_tree_search, breadth_first_tree_search, iterative_deepening_tree_search
from heuristics import blockworld_misplaced_tile_count, blockworld_manhattan, blockworld_manhattan_boost
from problemconcepts import Problem, State, Agent, SwitchTilesAction, Grid


def line_from_file_generator(file_name):

    with open(file_name) as text_file:
        for line in text_file:
            yield line

def build_grid_from_delimited_strings(delimited_strings, delimiter):

    grid_data = []
    for delimited_string in delimited_strings:
        grid_data.append(map(int, delimited_string.strip().split(delimiter)))

    return Grid(grid_data)


################################################# SET UP THE PROBLEM ################################################

# CREATE AGENT

def search_algorithm_wrapper(problem):
    #return iterative_deepening_tree_search(problem)
    return depth_first_tree_search(problem)
    #return breadth_first_tree_search(problem)
    #return a_star_tree_search(problem, blockworld_manhattan)

agent = Agent(name='Tile Switcher',
              skill=SwitchTilesAction,
              search_algorithm=search_algorithm_wrapper)

# CREATE INITIAL ENVIRONMENT AND STATE
lines_from_file = line_from_file_generator('blocksworld.txt')
environment = build_grid_from_delimited_strings(delimited_strings=lines_from_file, delimiter=' ')

# Define a pseudo-class for position
Position = namedtuple('Position', ['x', 'y'])

# Set agents initial position
agent_initial_position = Position(x=3, y=0)

# Create the tiles and goals
tiles = {'A': Position(0, 0),
         'B': Position(1, 0),
         'C': Position(2, 0),
         'D': Position(0, 1),
         'E': Position(1, 1),
         'F': Position(2, 1)
         }

tiles = {'A': Position(0, 0),
         'B': Position(1, 0),
         'C': Position(2, 0),
         'D': Position(0, 1),
         'E': Position(1, 1),
         }

goals = {'A': Position(1, 2),
         'B': Position(1, 1),
         'C': Position(1, 0),
         'D': Position(0, 2),
         'E': Position(0, 1),
         'F': Position(0, 0)
         }

global initial_state
initial_state = State(environment_data={'agent_location': agent_initial_position, 'tiles': tiles, 'goals': goals})

# CREATE A PROBLEM OBJECT

# Define the goal test
def goal_test_function(state):

    tiles = state.environment_data['tiles']
    goals = state.environment_data['goals']
    all_goals_met = True

    for tile_key, position in tiles.items():
        if position != goals[tile_key]:
            all_goals_met = False
            break

    return all_goals_met


problem = Problem(goal_test_function=goal_test_function,
                  initial_state=initial_state,
                  environment=environment,
                  agent=agent,
                  all_skills={SwitchTilesAction})


################################################# EXECUTE THE PROBLEM ################################################

# Execute search
num_iters_sample = []

for i in range(25):
    solution = problem.search()
    if solution[0] is not None:
        num_iters_sample.append(solution[1])
        print(solution[1])

print(num_iters_sample)

solution = problem.search()

# Print any successful action sequence with cost
if solution[0] is not None:
    goal_state, num_iterations, final_ply_depth, action_sequence = solution

    total_cost = 0
    state = initial_state
    moves = []
    i = 0
    for action in action_sequence:
        i += 1
        result = action.execute(state)

        x_change = result.state.environment_data['agent_location'].x - state.environment_data['agent_location'].x
        y_change = result.state.environment_data['agent_location'].y - state.environment_data['agent_location'].y

        if x_change == -1:
            moves.append("Left")
        elif x_change == 1:
            moves.append("Right")
        elif y_change == -1:
            moves.append("Down")
        elif y_change == 1:
            moves.append("Up")


        state = result.state
        tiles = state.environment_data['tiles']
        print("Action" , str(i), "\nMove to", action.agent_destination)
        for tile_key, position in tiles.items():
            print("Result:", tile_key, "at", tiles[tile_key])
        total_cost += result.cost

    print('Total cost', total_cost)
    print('Final ply depth: ', final_ply_depth)
    print('Action sequence len: ', len(action_sequence))
    print('Total nodes expanded: ', num_iterations)
    print('Action sequence works:', goal_test_function(state))
    print('Moves:', moves)

else:
    print('No solution found')
