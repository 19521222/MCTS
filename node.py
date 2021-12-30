import time
import chess
import chess.engine
import numpy as np
import concurrent.futures

from Chess_read import *
from env import *
from collections import defaultdict

engine = chess.engine.SimpleEngine.popen_uci(r"E:/Code/MCTS/stockfish_14.1_win_x64_avx2.exe")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(value):
    return 1 / (1 + np.power(10, -1*value / 50))

class Node:
    def __init__(self, state: ChessEnv, parent = None, move = None, policy = 'random_rollout'):
        self.state              = state
        self.parent             = parent
        self.move               = move
        self.children           = []
        self.nums_visits        = 0
        self.result             = 0
        self.unexplored_move    = None
        self.policy             = policy

    @property
    def Q(self):
        return self.result
    
    @property
    def N(self):
        return self.nums_visits

    @property
    def get_action(self):
        if self.unexplored_move is None:
            self.unexplored_move = self.state.generate_moves()
        return self.unexplored_move

    def is_fully_expanded(self):
        return self.unexplored_move == []

    def best_child(self, param = 1.414):
        weights = [(child.Q / child.N) + param * np.sqrt((np.log(self.N) / child.N)) for child in self.children]
        return self.children[np.argmax(weights)]

    def expand(self):
        move = np.random.choice(self.get_action)

        self.unexplored_move.remove(move)
        next_state  = self.state.take_action(move)
        child       = Node(next_state, parent = self, move = move)

        self.children.append(child)
        return child

    def is_terminate(self):
        return self.state.done

    def random_rollout(self, root_turn, gamma = 1):
        curr_state  = self.state
        depth       = 0
        turn        = -1 * root_turn if curr_state.board.turn else 1 * root_turn
        while not curr_state.done:
            if curr_state.ply > self.state.ply + 50:
                break
            legal_moves = curr_state.generate_moves()
            action      = np.random.choice(legal_moves)
            curr_state.step(action)
            depth       += 1
        
        result = curr_state._game_over()
        
        if result == 0.5:
            return result * (gamma ** depth)
        else:
            return turn * root_turn * result * (gamma ** depth)

    def negamax_rollout(self, root_turn, gamma = 1):
        curr_state  = self.state
        depth       = 0
        turn        = -1 * root_turn if curr_state.board.turn else 1 * root_turn
        for _ in range(2):
            if curr_state.done:
                break
            action   = curr_state.find_best_move(1)
            curr_state.step(action)
            depth   += 1

        while not curr_state.done:
            if curr_state.ply > self.state.ply + 50:
                break
            legal_moves = curr_state.generate_moves()
            action      = np.random.choice(legal_moves)
            curr_state.step(action)
            depth       += 1
        
        result = curr_state._game_over()
        if result == 0.5:
            return result * (gamma ** depth)
        else:
            return turn * root_turn * result * (gamma ** depth)

    def stockfish_rollout(self, root_turn, gamma = 1):
        curr_state  = self.state
        depth       = 0
        turn        = -1 * root_turn if curr_state.board.turn else 1 * root_turn
        while not curr_state.done:
            if curr_state.ply > self.state.ply + 50:
                break
            action  = engine.play(curr_state.board, chess.engine.Limit(time=0.0001))
            curr_state.step(action.move.uci())
            depth   += 1
        
        result = curr_state._game_over()
        if result == 0.5:
            return result * (gamma ** depth)
        else:
            return turn * root_turn * result * (gamma ** depth)

    def update(self, result, gamma = 1):
        self.nums_visits    += 1
        self.result         += result
        if result != 0.5:
            result = -result
        if self.parent:
            self.parent.update(gamma*result)

def random(node, root_turn, gamma = 0.99):
    return node.random_rollout(root_turn, gamma)

def negamax(node, root_turn, gamma = 0.99):
    return node.negamax_rollout(root_turn, gamma)

def stockfish(node, root_turn, gamma = 0.99):
    return node.stockfish_rollout(root_turn, gamma)

class MCTS:
    def __init__(self, policy = random, gamma = 0.99):
        self.policy         =  policy
        self.gamma          =  gamma

    def find_best_move(self, state, iter = 1000, max_time = None, detail = False):
        self.root           =  Node(state)
        root_turn = 1 if self.root.state.board.turn else -1
        if max_time:
            end_time    = time.time() + max_time
            iter        = 0
            while time.time() < end_time:
                iter    +=  1
                node    = self.stimulate()
                reward  = self.policy(node, root_turn, self.gamma)
                node.update(reward, self.gamma)
            print('Iter: ' ,iter)
        else:
            for _ in range(iter):
                node    = self.stimulate()
                reward  = self.policy(node, root_turn, self.gamma)
                node.update(reward)
        if detail:
            for child in self.root.children:
                print(child.move, child.Q / child.N)
        
        best_node = self.root.best_child(param = 0)
        return best_node.move

    def stimulate(self):
        curr_node = self.root
        while not curr_node.is_terminate():
            if not curr_node.is_fully_expanded():
                return curr_node.expand()
            else:
                curr_node = curr_node.best_child()
        return curr_node

