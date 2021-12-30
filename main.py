from env import *
from node import *
from NN_Player import *
from argparse import ArgumentParser

class Agent:
    def __init__(self, max_time, max_depth = 4, agent = "human", side = 1):
        self.side       = side
        self.agent      = agent
        self.max_time   = max_time
        self.max_depth  = max_depth
    
    def best_move(self, env):
        if self.agent == "human":
            move = input("Move: ")
            return move
        else:
            return self.agent.find_best_move(env, max_time = self.max_time, detail = False)


def main():
    arg = ArgumentParser()
    arg.add_argument("--mode", help="Self-Play or Versus", default="self", choices=["self", "vs"])
    arg.add_argument("--option", help="Type of MCTS", default="random", choices=["random", "minimax", "stockfish", "nn"])
    arg.add_argument("--time", help="Time to run MCTS", type=int,  default=30)
    arg.add_argument("--fen", help="Custom FEN")
    args = arg.parse_args()
    
    env = ChessEnv()
    if args.fen:
        env.update(args.fen)
    else:
        env.reset()

    mode        = 1 if args.mode == "self" else 0
    if args.option == "random":
        search  = MCTS()
    elif args.option == "minimax":
        search  = MCTS(negamax)
    else:
        search  = MCTS(stockfish)
    time_limit  = args.time
    env.render()

    side        = None
    white       = None
    black       = None
    if mode == 0:
        side        = input("White?: 1 or 0")
    if side is not None:
        if side == 1:
            white       = Agent(time_limit, agent="human")
            black       = Agent(time_limit, agent=search)
        else:
            black       = Agent(time_limit, agent="human")
            white       = Agent(time_limit, agent=search)
    else:
            white       = Agent(time_limit, agent=search)
            black       = Agent(time_limit, agent=search)


    while not env.done:
        move            = white.best_move(env.copy())
        print('Move: ', move)
        env.step(move)

        env.render()        
    
        move            = black.best_move(env.copy())
        print('Move: ', move)
        env.step(move)

        env.render()

    
    print("End Game")
    exit()

if __name__ == "__main__":
    main()