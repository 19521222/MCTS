import chess
import copy
import numpy as np
from chess import UNICODE_PIECE_SYMBOLS
from heuristics import *


killer_moves = np.empty((2, 128), dtype= chess.Move)
history_moves = np.zeros((12, 64), dtype= chess.Move)
pv_len = [0] * 128
pv_table = np.empty((128, 128), dtype=chess.Move)

def reset():
    killer_moves = np.empty((2, 128), dtype= chess.Move)
    history_moves = np.zeros((12, 64), dtype= chess.Move)
    pv_len = [0] * 128
    #pv_table = empty((128, 128), dtype=chess.Move)

def coord_from_pos(num):
    return num // 8, num % 8

class ChessEnv:
    """
    Represents a chess environment where a chess game is played/
    Attributes:
        :ivar chess.Board board: current board state
        :ivar int ply: number of half moves performed in total by each player
        :ivar Winner winner: winner of the game
        :ivar boolean resigned: whether non-winner resigned
        :ivar str result: str encoding of the result, 1-0, 0-1, or 1/2-1/2
    """
    def __init__(self):
        self.board = None
        self.resigned = False
        self.result = None
        self.legal_moves = None

    def reset(self):
        """
        Resets to begin a new game
        :return ChessEnv: self
        """
        self.board = chess.Board()
        self.legal_moves = list(self.board.legal_moves)
        self.winner = None
        self.resigned = False
        return self

    def update(self, board):
        """
        Like reset, but resets the position to whatever was supplied for board
        :param chess.Board board: position to reset to
        :return ChessEnv: self
        """
        self.board = chess.Board(board)
        self.legal_moves = list(self.board.legal_moves)
        self.winner = None
        self.resigned = False
        return self

    @property
    def  ply(self):
        return self.board.ply()

    @property
    def final_result(self):
        return self.result

    @property
    def done(self):
        return self.board.is_game_over()

    @property
    def white_to_move(self):
        return self.board.turn == chess.WHITE

    @property
    def observation(self):
        return self.board.fen()

    def generate_moves(self):
        return list(map(str, self.board.legal_moves))

    def copy(self):
        env = copy.deepcopy(self)
        env.board = copy.deepcopy(self.board)
        return env

    def take_action(self, action: str):
        new_env = self.copy()
        new_env.board.push_uci(action)
        return new_env

    def _game_over(self):
        self.result = self.board.result(claim_draw = True)
        if self.result == '1-0':
            self.winner = 'WHITE'
            return 1
        elif self.result == '0-1':
            self.winner = 'BLACK'
            return -1
        else:
            self.winner = 'DRAW'
            return 0.5

    def step(self, action: str, check_over = True):
        """
        Takes an action and updates the game state
        :param str action: action to take in uci notation
        :param boolean check_over: whether to check if game is over
        """
        if check_over and action is None:
            self._resign()
            return

        self.board.push_uci(action)

        if check_over and self.board.result(claim_draw=True) != "*":
            self._game_over()
        self.legal_moves = list(self.board.legal_moves)

    def render(self):
        #print(self.board.unicode())
        fen = self.board.fen().split(" ")[0].split("/")
        print("\n")
        print("  A B C D E F G H")
        offset = 8
        for row in fen:
            print(offset, end=' ')
            for c in row:
                if c.isalpha():
                    print(UNICODE_PIECE_SYMBOLS[c], end=' ')
                else:
                    print('. ' * int(c), end='')
            offset -= 1
            print()
        print("\n")

    def evaluate(self, side):
        map = self.board.piece_map()
        keys = list(map.keys())

        score = 0
        for key in keys:
            y, x = coord_from_pos(key)
            c = map[key].symbol()
            if c == "P":
                score += 1 + pawnEvalWhite[y][x]
            elif c == "B":
                score += 10 + bishopEvalWhite[y][x]
                attack = self.board.attacks(chess.square(x, y)).tolist()
                score += sum(np.logical_and(attack, occupancies[0]))
            elif c == "N":
                score += 8 + knightEval[y][x]
            elif c == "R":
                score += 15 + rookEvalWhite[y][x]
            elif c == "Q":
                score += 30 + queenEval[y][x]
                attack = self.board.attacks(chess.square(x, y)).tolist()
                score += sum(np.logical_and(attack, occupancies[0]))
            elif c == "K":
                score += 200 + kingEvalWhite[y][x]
                attack = self.board.attacks(chess.square(x, y)).tolist()
                score += 5 * sum(np.logical_and(attack, occupancies[2]))
            elif c == "p":
                score -= (1 + pawnEvalBlack[y][x])
            elif c == "b":
                score -= (10 + bishopEvalBlack[y][x])
                attack = self.board.attacks(chess.square(x, y)).tolist()
                score -= sum(np.logical_and(attack, occupancies[0]))
            elif c == "n":
                score -= (8 + knightEval[y][x])
            elif c == "r":
                score -= (15 + rookEvalBlack[y][x])
            elif c == "q":
                score -= (30 + queenEval[y][x])
                attack = self.board.attacks(chess.square(x, y)).tolist()
                score -= sum(np.logical_and(attack, occupancies[0]))
            elif c == "k":
                score -= (200 + kingEvalBlack[y][x])
                attack = self.board.attacks(chess.square(x, y)).tolist()
                score -= 5 * sum(np.logical_and(attack, occupancies[1]))
        
        return score if side else -score

    def quiescence(self, alpha = -100000, beta = 100000):
              
        eval = self.evaluate(self.board.turn)
        
        if eval >= beta:
            return beta

        if eval >  alpha:
            alpha = eval

        moves = self.legal_moves
        moves.sort(reverse=True, key=self.score_move)

        for move in moves:
            if self.board.is_capture(move):
                try:
                    new_env = self.copy()
                    new_env.step(move.uci())
                    nodeVal = - (new_env.quiescence(-beta, -alpha))
                except:
                    continue
                if nodeVal >= beta:
                    return beta
                if nodeVal > alpha:
                    alpha = nodeVal
        return alpha
    
    def negamax(self, depth = 2, alpha = -100000, beta = 100000):
        pv_len[self.ply] = self.ply

        #The Ply out of range
        if self.ply >= 128:
            return self.evaluate(self.board.turn)

        if depth == 0:
            return self.evaluate(self.board.turn)
            #return self.quiescence(alpha, beta)

        checked = self.board.is_check()
        if checked:
            depth += 1

        if depth >= 3 and not checked and self.ply != 0:
            new_env = self.copy()
            new_env.step('0000')
            nodeVal = - new_env.negamax(depth - 1 - 2, -beta, -beta + 1)

            if nodeVal >= beta:
                return beta

        moves = list(self.board.legal_moves)
        move_searched = 0
        #PV_node
        self.follow_pv = 0
        if pv_table[0][self.ply] in moves:
            self.follow_pv = 1
            self.score_pv = 1

        moves.sort(reverse=True, key=self.score_move)
        for move in moves:
            new_env = self.copy()
            new_env.step(move.uci())
            try:
                new_env = self.copy()
                new_env.step(move.uci())
            except:
                continue
            #Move Ordering
            #1. Principle Variation
            #2. 1st Killer Move
            #3. 2nd Killer Move
            #4. History Move
            #5. Unodering Move
            if move_searched == 0:
                #Principle Variation Search
                nodeVal = -(new_env.negamax(depth - 1, -beta, -alpha))
            else:
                # Done searching the top-4 move ordering then we go for LMR
                # When depth >= 3 LMR get in to the search
                if move_searched >= 4 and depth >= 3 and not checked and not self.board.is_capture(move):
                    #Search with less depth to determine whether the move is good or bad (LMR)
                    nodeVal = - (new_env.negamax(depth - 2, -alpha - 1, -alpha))
                else:
                    #Trick to activate the next condition line
                    nodeVal = alpha + 1

                if nodeVal > alpha:
                    #If current move is a good move, research it with full-depth and narrow window
                    nodeVal = - (new_env.negamax(depth - 1, -alpha - 1, -alpha))
                    #If still failed, do a normal search
                    if nodeVal > alpha and nodeVal < beta:
                        nodeVal = -(new_env.negamax(depth - 1, -beta, -alpha))
            
            move_searched += 1

            if nodeVal >= beta:
                if not self.board.is_capture(move):
                    #If the move is fail high and be a non_capture move
                    #Store its as killer move
                    killer_moves[1][self.ply] = killer_moves[0][self.ply]
                    killer_moves[0][self.ply] = move
                return beta
            if nodeVal > alpha:
                if not self.board.is_capture(move):
                    #If the move is pricinple move and be a non_capture move
                    #Store its as history move
                    f = self.board.piece_type_at(move.from_square)
                    t = move.to_square
                    history_moves[f-1][t-1] += depth
                alpha = nodeVal
                #Store the best move as Principle Variation Move for each ply
                pv_table[self.ply][self.ply] = move
                for next in range(self.ply+1, pv_len[self.ply+1]):
                    pv_table[self.ply][next] = pv_table[self.ply + 1][next]
                pv_len[self.ply] = pv_len[self.ply + 1]
        return alpha

    def score_move(self, move: chess.Move):
        if self.score_pv:
            if pv_table[0][self.ply] == move:
                self.score_pv = 0
                return 20000

        if self.board.is_capture(move):
            f = self.board.piece_type_at(move.from_square)
            t = self.board.piece_type_at(move.to_square)
            try:
                return mvv_lva[f-1][t-1] + 10000
            except:
                return 105 + 10000
        else:
            if killer_moves[0][self.ply] == move:
                return 9000
            elif killer_moves[1][self.ply] == move:
                return 8000
            else:
                f = self.board.piece_type_at(move.from_square)
                t = move.to_square
                if f != None and t != None:
                    return history_moves[f-1][t-1]

        return 0

    def find_best_move(self, depth):
        self.follow_pv = 1
        self.score_pv = 0
        window_val = 50
        alpha = -100000
        beta = 100000
        reset()

        for d in range(1, depth + 1):
            self.follow_pv = 1

            #start = time.time()
            score_move = self.negamax(d, alpha, beta)
            # print('DEPTH:', d)
            # print("Time:", time.time() - start)
            # print("Computer Move: {}, Node: {}, Eval: {}".format(
            #     pv_table[self.ply][self.ply].uci(), 
            #     NODE, 
            #     score_move))

            # for move in pv_table[self.ply]:
            #     if move != None:
            #         print(move, end= ' ')

            if score_move <= alpha or score_move >= beta:
                alpha = -100000
                beta = 100000
                continue

            alpha = score_move - window_val
            beta = score_move + 50

        # if self.ply >= 100:
        #     if score_move < -40 and score_move > 40:
        #         self._resign()
        #         print('{} resign.'.format(self.board.turn))
        #         print('Winner:', self.winner)
        #         return

        return pv_table[self.ply][self.ply].uci()