from maze import *
from mcts import *


def dec_demo():
    targets = {(1,7): 2, (1,8):2, (1,9):2,
               (2,7): 2, (2,8):3, (2,9):2,
               (3,7): 2, (3,8):2, (3,9):2,
               (7,1): 2, (8,1):2, (9,1):2,
               (7,2): 2, (8,2):3, (9,2):2,
               (7,3): 2, (8,3):2, (9,3):2,}
    obstacles = {(5,6), (6,5), (4,5), (5,4), (5,5)}
    env = MazeEnvironment(xlim=(0, 10), ylim=(0, 10), obstacles=obstacles,
                          targets=targets, is_border_obstacle_filled=True)
    state = MazeState(environment=env, time_remains=11)
    state.add_agent((1, 1))
    state.add_collaborator(*gen_collaborator())
    return state


def gen_1x1_kernel(center_pos):
    return {center_pos: 1}

def gen_3x3_kernel(center_pos):
    center_x, center_y = center_pos
    kernel = {}
    height = 1
    row    = range(-height,height+1)
    col    = range(-height,height+1)
    base   = [[1, 2, 1],
              [2, 4, 2],
              [1, 2, 1]]
    for j in range(len(row)):
        for i in range(len(col)):
            kernel[(center_x+row[i], center_y+row[j])] = base[j][i]
    return kernel

def gen_5x5_kernel(center_pos):
    center_x, center_y = center_pos
    kernel = {}
    height = 2
    row    = range(-height,height+1)
    col    = range(-height,height+1)
    base   = [[1,  4,  7,  4,  1],
              [4, 16, 26, 16,  4],
              [7, 26, 41, 26,  7],
              [4, 16, 26, 16,  4],
              [1,  4,  7,  4,  1]]
    for j in range(len(row)):
        for i in range(len(col)):
            kernel[(center_x+row[i], center_y+row[j])] = base[j][i]
    return kernel


def gen_7x7_kernel(center_pos):
    center_x, center_y = center_pos
    kernel = {}
    height = 3
    row    = range(-height,height+1)
    col    = range(-height,height+1)
    base   = [[0,  0,  1,   2,  1,  0,  0],
              [0,  3, 13,  22, 13,  3,  0],
              [1, 13, 59,  97, 59, 13,  1],
              [2, 22, 97, 159, 97, 22,  2],
              [1, 13, 59,  97, 59, 13,  1],
              [0,  3, 13,  22, 13,  3,  0],
              [0,  0,  1,   2,  1,  0,  0]]
    for j in range(len(row)):
        for i in range(len(col)):
            kernel[(center_x+row[i], center_y+row[j])] = base[j][i]
    return kernel


def gen_collaborator():
    action = [(9,9),(8,9),(7,9),(6,9),(6,8),
              (5,8),(4,8),(3,8),(2,8),(1,8),
              (1,7),(2,7)]
    sigma  = [gen_1x1_kernel((9,9)),
              gen_1x1_kernel((8,9)),
              gen_1x1_kernel((7,9)),
              gen_3x3_kernel((6,9)),
              gen_3x3_kernel((6,8)),
              gen_3x3_kernel((5,8)),
              gen_5x5_kernel((4,8)),
              gen_5x5_kernel((3,8)),
              gen_5x5_kernel((2,8)),
              gen_7x7_kernel((1,8)),
              gen_7x7_kernel((1,7)),
              gen_7x7_kernel((2,7))]

    return action, sigma


def simulate(mcts_select_policy, mcts_expand_policy, mcts_rollout_policy,
             mcts_backpropagate_policy, initial_state: MazeState,
             rand_seed: int = 0) -> MazeState:
    mcts = MonteCarloSearchTree(initial_state, max_tree_depth=11, samples=2000,
                                tree_select_policy=mcts_select_policy,
                                tree_expand_policy=mcts_expand_policy,
                                rollout_policy=mcts_rollout_policy,
                                backpropagate_method=mcts_backpropagate_policy)
    random.seed(rand_seed)
    state = initial_state.__copy__()
    time = 0
    while not state.is_terminal:
        actions = mcts.search_for_actions(search_depth=11)
        time += 1
        print("Time step {0}".format(time))
        for i in range(len(state.paths)):
            action = actions[i]
            print(action)
            state = state.execute_action(action)
            mcts.update_root(action)
            if state.is_terminal:
                break
    return state


if __name__ == "__main__":
    print("===== Start of Decentralized-MCTS Demo =====")
    simulate(initial_state=dec_demo(),
             mcts_select_policy=select,
             mcts_expand_policy=expand,
             mcts_rollout_policy=default_rollout_policy,
             mcts_backpropagate_policy=backpropagate).visualize()

# IMPROVEMENTS 
# TODO make a new branch and double check that everything is working before committing
# TODO delete TODO comments 