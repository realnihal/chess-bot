import random

class DummyEngine(object):

    def __init__(self):
        None
    
    def play(self, board):
        moves = list(board.legal_moves)
        
        return random.choice(moves)