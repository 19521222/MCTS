import numpy as np
import chess

class CHESS_READ():
    def __init__(self):
        #states
        # 0 - empty; 1 - King; 2 - Queen; 3 - Bishop; 4 - Knight; 5 - Rook; 6 - Pawn;  7 - king; 8 - queen; 9 - bishop; 10 - knight; 11 - rook; 12 - pawn;
        self.piece        = ['.', 'K', 'Q', 'B', 'N', 'R', 'P', 'k', 'q', 'b', 'n', 'r', 'p']
        
        #actions
        self.action_space = []
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
        promoted_to = ['q', 'r', 'b', 'n']

        for l1 in range(8):
            for n1 in range(8):
                #slide horizontal
                #slide vertical
                #slide 2 diagonal lines
                #knight move
                destinations = [
                    (t, n1) for t in range(8)] + \
                    [(l1, t) for t in range(8)] + \
                    [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                    [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                    [(l1 + a, n1 + b) for (a, b) in [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]
                ]
                for (l2, n2) in destinations:
                    if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                        move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                        self.action_space.append(move)
        #promotion move
        for l1 in range(8):
            l = letters[l1]
            for p in promoted_to:
                self.action_space.append(l + '2' + l + '1' + p)
                self.action_space.append(l + '7' + l + '8' + p)
                if l1 > 0:
                    l_l = letters[l1 - 1]
                    self.action_space.append(l + '2' + l_l + '1' + p)
                    self.action_space.append(l + '7' + l_l + '8' + p)
                if l1 < 7:
                    l_r = letters[l1 + 1]
                    self.action_space.append(l + '2' + l_r + '1' + p)
                    self.action_space.append(l + '7' + l_r + '8' + p)
        
    def read_state(self, board):
        # convert board information from Chess into 8x8 numpy  
        state_tmp = [self.piece.index(item) for item in str(board) if item != ' ' and item !='\n']
        state_tmp = np.array(state_tmp).reshape(8,8)   
        state     = np.zeros((13,8,8))
        for i in range(8):
            for j in range(8):
                state[state_tmp[i,j],i,j] = 1.0         # assign probabiliry 1.0 to each piece in square
        return state
    
    def read_move(self, idx):
        return chess.Move.from_uci(self.action_space[idx])
    
    def read_idx(self, move): 
        move = str(move)
        return self.action_space.index(move)