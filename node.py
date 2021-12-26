import time
import concurrent.futures
from chess import Move
import numpy as np
from env import *
from collections import defaultdict

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class Node:
    def __init__(self, state: ChessEnv, parent = None, move = None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.nums_visits = 0
        self.result = defaultdict(int)
        self.unexplored_move = None

    @property
    def Q(self):
        wins = self.result[1]
        loses = self.result[-1]
        draw = self.result[0.5]
        return wins - loses + draw
    
    @property
    def N(self):
        return self.nums_visits

    @property
    def get_action(self):
        if self.unexplored_move is None:
            self.unexplored_move = self.state.generate_moves()
        return self.unexplored_move
    
    def score_move(self, move):
        b_val = self.state.evaluate(True)
        self.state.board.push_uci(move)
        a_val = self.state.evaluate(True)
        self.state.board.pop()
        return np.abs(a_val - b_val)


    def is_fully_expanded(self):
        return self.unexplored_move == []

    def best_child(self, param = 1.4):
        weights = [(child.Q / child.N) + param * np.sqrt((2 * np.log(self.N) / child.N)) for child in self.children]
        return self.children[np.argmax(weights)]

    def expand(self):
        move = np.random.choice(self.get_action)
        # moves = self.get_action
        # moves.sort(key=self.score_move)
        # move = moves[-1]
        self.unexplored_move.remove(move)
        next_state = self.state.take_action(move)
        child = Node(next_state, parent = self, move = move)

        self.children.append(child)
        return child

    def is_terminate(self):
        return self.state.done

    def random_rollout(self):
        curr_state = self.state
        #turn = 1 if curr_state.board.turn else -1
        turn = curr_state.board.turn
        while not curr_state.done:
            if curr_state.ply > self.state.ply  + 100:
                break
            legal_moves = curr_state.generate_moves()
            action = np.random.choice(legal_moves)
            curr_state = curr_state.take_action(action)
        
        result = curr_state._game_over() 
        if result == 0.5:
            return result
        else:
            return turn * result

    def negamax_rollout(self):
        curr_state = self.state
        turn = 1 if curr_state.board.turn else -1

        for _ in range(3):
            if curr_state.done:
                break
            action = curr_state.find_best_move(2)
            curr_state = curr_state.take_action(action)

        while not curr_state.done:
            if curr_state.ply > self.state.ply  + 100:
                break
            legal_moves = curr_state.generate_moves()
            action = np.random.choice(legal_moves)
            curr_state = curr_state.take_action(action)
        
        result = curr_state._game_over() 
        if result == 0.5:
            return result
        else:
            return turn * result

    def update(self, result):
        self.nums_visits += 1
        self.result[result] += 1
        if self.parent:
            self.parent.update(-result)

class MCTS:
    def __init__(self, state):
        self.root = Node(state)

    def best_action(self, iter = 200, max_time = None):
        if max_time:
            end_time = time.time() + max_time
            iter = 0
            while time.time() < end_time:
                iter +=  1
                node = self.stimulate()
                reward = node.negamax_rollout()
                #reward = node.random_rollout()
                node.update(reward)
                # with concurrent.futures.ThreadPoolExecutor() as executor:
                #     futures = []
                #     for _ in range(4):
                #         futures.append(executor.submit(rollout, node))
                #     for future in concurrent.futures.as_completed(futures):
                #         reward = future.result()
                #         node.update(reward)
            print('Iter: ' ,iter)
        else:
            for _ in range(iter):
                node = self.stimulate()
                reward = node.negamax_rollout()
                node.update(reward)
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

def rollout(node: Node):
    return node.negamax_rollout()