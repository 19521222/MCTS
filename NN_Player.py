from Chess_read import *
from Neural_Network import *
from NN_MCTS import *

from torchviz.dot import make_dot

class PLAYER():
    def __init__(self):
        self.device = 'cpu' 
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = "best_policy.model"
        
        self.mcts_iteration = 100
        self.mcts_time      = 30
        self.learning_rate  = 0.001
        
        self.Chess_read     = CHESS_READ()
        self.NN             = NEURAL_NETWORK(self.learning_rate, self.path, self.device)
        
        #Load saved model
        self.NN.load_model()
        self.mcts_tree      = NN_MCTS(self.mcts_iteration, self.mcts_time, self.NN, self.Chess_read, self.device)
    
    def visualize(self):
        x                   = Variable(torch.Tensor(self.Chess_read.read_state(chess.Board()).reshape(1,13,8,8))).to(self.device)
        make_dot(self.NN.policy_net(x), params=dict(self.NN.policy_net.named_parameters())).render("model", format = 'png')

    def test(self): 
        board      = chess.Board()
        move       = None
        print(board.unicode(), end='\n')
        while not board.is_game_over():
            print(move)
            self.mcts_tree.simulation(board, move)
            node_tmp = self.mcts_tree.node_list[str(board), move]
            if str(node_tmp.board) != str(board):
                print('warning: node does not match board')
                break
                
            move, prob = self.mcts_tree.play(board, move)   # False - no stockfish; True - run stockfish
            if move not in list(board.legal_moves): 
                print(move)
                print('warning: move is not legal')
                break
            if round(sum(prob),1) != 1.0:
                print('warning: Pi.sum()!= 1')
                break
            board.push(move)
            print(board.unicode(), end='\n')
        return board.is_game_over()