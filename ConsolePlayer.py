import chess

class ConsolePlayer(object):

    def __init__(self):
        None
    
    def play(self,board):
        while True:
            try:
                m = input("Enter your move (e.g., e2e4):")
            except ValueError:
                print("Invalid Move")
                m = None
            move = chess.Move.from_uci(m)
            if board.is_legal(move):
                break
            print("Illegal move.")

        return move
