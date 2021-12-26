from node import *


def main():
    env = ChessEnv()
    env.reset()
    env.render()
    while not env.done:
        search = MCTS(env)
        move = search.best_action(max_time=30)
        env.step(move)
        env.render()

main()