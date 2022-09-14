import chess
import time
from math import inf as INFINITY
from collections import namedtuple

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_count(new_board):
    # count material in the new position for player who just moved

    if new_board.is_stalemate():
        return 0

    all_pieces = new_board.piece_map().values()

    material_difference = 0

    for piece in all_pieces:
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == new_board.turn:
            material_difference -= value
        else:
            material_difference += value

    if new_board.is_checkmate():
        material_difference += 999999

    return material_difference


def improved_score(new_board):
    score = material_count(new_board)

    # add extra score strategies

    # compute space controlled by player who just moved

    space = 0
    for square in chess.SQUARES:
        if new_board.is_attacked_by(not new_board.turn, square):
            space += 1
        if new_board.is_attacked_by(new_board.turn, square):
            space -= 1

    score += space * 1/64

    # # remove hanging pieces from material count
    #
    # all_pieces = new_board.piece_map().items()
    #
    # for square, piece in all_pieces:
    #     if piece.color == turn:
    #         attacker_count = len(new_board.attackers(not turn, square))
    #         defender_count = len(new_board.attackers(turn, square))
    #         if attacker_count > defender_count:
    #             score -= PIECE_VALUES[piece.symbol().upper()]

    return score


num_pruned = 0
cache_hits = 0
positions = 0


Config = namedtuple("Config",
                    ['prune', 'cache', 'sort', 'max_depth', 'sort_heuristic'],
                    defaults=[True, True, True, 4, None])


def minimax_score(board, opponent_best=INFINITY, my_best=-INFINITY, curr_depth=0,
                  cache=(), config=Config(), sort_heuristic=material_count):

    global cache_hits, num_pruned, positions

    positions += 1

    turn = board.turn

    if curr_depth == config.max_depth or board.outcome():
        return improved_score(board)

    # recursively reason about best move

    moves = list(board.legal_moves)
    best_move = None
    best_score = -INFINITY
    
    children = []

    # generate children positions from legal moves
    for move in moves:
        # apply the current candidate move

        new_board = board.copy()
        new_board.push(move)
        
        sort_score = sort_heuristic(new_board) \
            if config.sort else 0

        children.append((sort_score, new_board, move))

    for _, new_board, move in sorted(children, key=lambda x: x[0], reverse=True):

        if config.cache:
            # The cache saves score and depth of score calculation.

            fen = new_board.fen()
            fen = fen[:-4]  # remove move counts from fen

            score, cached_depth = cache[fen] if fen in cache else (0, 0)

            # depth of score estimate if we compute it
            new_depth = config.max_depth - curr_depth

            # if we could get a deeper estimate than what is in the cache
            if new_depth > cached_depth:
                score = minimax_score(new_board, -my_best, -opponent_best, curr_depth + 1, cache, config, sort_heuristic)

                cache[fen] = (score, new_depth)
            else:
                cache_hits += 1
        else:
            score = minimax_score(new_board, -my_best, -opponent_best, curr_depth + 1, cache, config, sort_heuristic)

        if score > best_score:
            best_move = move
            best_score = score
            my_best = max(best_score, my_best)

        if config.prune:
            if score > opponent_best:
                num_pruned += 1
                return -best_score

    # print("Opponent's best move is {}".format(best_move))

    return -best_score


class ScoreEngine(object):

    def __init__(self, score_function = minimax_score, name=None, config=Config()):
        self.name = name
        self.score_function = score_function
        self.config = config
        self.known_positions = {}

    def cached_score(self, new_board):
        fen = new_board.fen()
        fen = fen[:-4]  # remove move counts from fen
        # todo: refactor to create standard modified FEN

        if fen in self.known_positions:
            score, _ = self.known_positions[fen]
            return score
        return material_count(new_board)

    def play(self, board):
        start_time = time.time()
        moves = list(board.legal_moves)

        best_move = None
        best_score = -INFINITY # simulating -ve infinty
        for move in moves:
            # apply current candidate move to the board 
            new_board = board.copy()
            new_board.push(move)
            # count material score on the board
            

            score = self.score_function(new_board, cache=self.known_positions,
                                        config=self.config, curr_depth=1,
                                        sort_heuristic=self.cached_score)
                
            if score > best_score:
                best_move = move
                best_score = score
        
        print("Found move in {} seconds".format(time.time() - start_time))
        return best_move


if __name__ == "__main__":
    board = chess.Board('8/5Qpk/B4bnp/8/3r4/PR4PK/1P3P1P/6r1 b - - 2 31')
    # board = chess.Board('3rk3/1p2qp2/2p2n2/1B3bp1/1b1Qp3/8/PPPP1PP1/RNB1K1N1 w Q - 0 23')
    # board = chess.Board('Q1R5/6K1/1k6/3B4/5r1P/5rP1/8/1r6 b - - 0 1')

    configs = [Config(sort_heuristic=material_count, max_depth=4),
               Config(sort_heuristic=False, max_depth=4)
               ]

    for config in configs:
        # todo: refactor to keep stats without global variables
        cache_hits = 0
        num_pruned = 0
        positions = 0

        print("Starting " + repr(config))

        # todo: very ugly hack to get some quick experiments
        # configuration currently does not consider the sort heuristic.
        # it's hard coded to use cached score.
        if not config.sort_heuristic:
            config = Config(sort_heuristic=engine.cached_score, max_depth=4)

        engine = ScoreEngine(config=config)

        start_time = time.time()
        move = engine.play(board)
        print("Found move in {} seconds".format(time.time() - start_time))

        print("Cache hits: {}. Prunes: {}. Positions: {}.".format(cache_hits, num_pruned, positions))

        print(move)

        print("\n\n\n")


