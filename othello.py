import numpy as np
import copy
import sys
from time import time

NUM_COLUMNS = 8
# With these constant values for players, flipping ownership is just a sign change
WHITE = 1
NOBODY = 0
BLACK = -1

TIE = 2

WIN_VAL = 100
WHITE_TO_PLAY = True
DEMO_SEARCH_DEPTH = 5

# We'll sometimes iterate over this to look in all 8 directions from a particular square.
# The values are the "delta" differences in row, col from the original square.
# (Hence no (0,0), which would be the same square.)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1,1)]

# read_board returns an integer array representing a board position,
# using W, B, and - for white, black, and blank squares.  No delimiters,
# just one character per square
def read_board():
    board = np.zeros((NUM_COLUMNS,NUM_COLUMNS))
    board_chars = {
        'W': WHITE,
        'B': BLACK,
        '-': NOBODY
    }
    r = 0  # row
    for line in sys.stdin:
        for c in range(NUM_COLUMNS):
            board[r][c] = board_chars.get(line[c], NOBODY) # quietly ignore bad chars
        r += 1
    return board

# Return identity of winner, assuming game is over.
# Returns WHITE, BLACK, or TIE.
def find_winner(board):
    # Slick counting of values:  np.count_nonzero counts vals > 0, so pass in
    # board == WHITE to get 1 or 0 in the right spots
    white_count = np.count_nonzero(board == WHITE)
    black_count = np.count_nonzero(board == BLACK)
    if white_count > black_count:
        return WHITE
    elif white_count < black_count:
        return BLACK
    return TIE

# generate_legal_moves returns a list of (row, col) tuples representing places to move.
# Takes a board and whose turn it is (the moves are for that player) -- white_turn
# is a bool, True for white to go and False for black
def generate_legal_moves(board, white_turn):
    legal_moves = []
    for r in range(NUM_COLUMNS):
        for c in range(NUM_COLUMNS):
            if board[r][c] != NOBODY:
                continue   # Occupied, so not legal for a move
            # Legal moves must capture something
            if can_capture(board, r, c, white_turn):
                legal_moves.append((r,c))
    return legal_moves

# Helper that checks capture in each of 8 directions
def can_capture(board, r, c, white_turn):
    for r_delta, c_delta in DIRECTIONS:
        if captures_in_dir(board, r, r_delta, c, c_delta, white_turn):
            return True
    return False

# Check for capture in direction that modifies row by row_delta, col by col_delta
def captures_in_dir(board, row, row_delta, col, col_delta, white_turn):
    # Can't capture if headed off the board
    if (row+row_delta < 0) or (row+row_delta >= NUM_COLUMNS):
        return False
    if (col+col_delta < 0) or (col+col_delta >= NUM_COLUMNS):
        return False

    # Can't capture if piece in that direction is not of appropriate color or missing
    enemy_color = BLACK if white_turn else WHITE
    if board[row+row_delta][col+col_delta] != enemy_color:
        return False

    # At least one enemy piece in this direction, so just need to scan until we
    # find a friendly piece (return True) or hit an empty spot or edge of board
    # (return False)
    friendly_color = WHITE if white_turn else BLACK
    scan_row = row + 2*row_delta # row of first scan position
    scan_col = col + 2*col_delta # col of first scan position
    while (scan_row >= 0) and (scan_row < NUM_COLUMNS) and (scan_col >= 0) and (scan_col < NUM_COLUMNS):
        if board[scan_row][scan_col] == NOBODY:
            return False
        if board[scan_row][scan_col] == friendly_color:
            return True
        scan_row += row_delta
        scan_col += col_delta
    return False

# Destructively change a board to represent capturing a piece with a move at (row,col).
# The board's already a copy made specifically for the purpose of representing this move, 
# so there's no point in copying it again.  We'll return the board anyway.
def capture(board, row, col, white_turn):
    # Check in each direction as to whether flips can happen -- if they can, start flipping
    enemy_color = BLACK if white_turn else WHITE
    for row_delta, col_delta in DIRECTIONS:
        if captures_in_dir(board, row, row_delta, col, col_delta, white_turn):
            flip_row = row + row_delta
            flip_col = col + col_delta
            while board[flip_row][flip_col] == enemy_color:
                board[flip_row][flip_col] = -enemy_color
                flip_row += row_delta
                flip_col += col_delta
    return board

# play handles the logic of actually executing a move, returning the new board
# that results after putting down the new piece and flipping captured pieces.
# The board that is returned is a copy, so this is appropriate to use for search.
# move is a tuple (row, col) with coordinates for the move.
def play_move(board, move, white_turn):
    new_board = copy.deepcopy(board)
    new_board[move[0]][move[1]] = WHITE if white_turn else BLACK
    new_board = capture(new_board, move[0], move[1], white_turn)
    return new_board

# check_game_over returns the current winner of the board - WHITE, BLACK, TIE, NOBODY
def check_game_over(board):
    # It's not over if either player still has legal moves
    white_legal_moves = generate_legal_moves(board, True)
    if white_legal_moves:  # Python idiom for checking for empty list
        return NOBODY
    black_legal_moves = generate_legal_moves(board, False)
    if black_legal_moves:
        return NOBODY
    # I guess the game's over
    return find_winner(board)

# minimax_value:  assumes white is MAX and black is MIN (even if black uses this function)
# white_turn determines who would get to move next
# search depth is the depth remaining, decremented for recursive calls
# alpha and beta are the bounds on viable play values used in alpha-beta pruning
def minimax_value(board, white_turn, search_depth, alpha, beta):
    """
    Great explanation and diagrams in this video: https://www.youtube.com/watch?v=l-hh51ncgDI
    """
    if search_depth == 0:
        return np.count_nonzero(board == WHITE) - np.count_nonzero(board == BLACK)
    state = check_game_over(board)
    if state == TIE:
        return 0
    if state == WHITE:
        return WIN_VAL
    if state == BLACK:
        return -WIN_VAL
    else:
        pass # No game condition met, continue searching down the decision tree.
    
    if white_turn:
        maxScore = -sys.maxsize
        legal_moves = generate_legal_moves(board, True)

        if len(legal_moves) == 0:
            maxScore = minimax_value(board, False, search_depth, alpha, beta)

        else:
            for move in legal_moves:
                new_board = play_move(board, move, True) # Get the new board state
                score = minimax_value(new_board, False, search_depth - 1, alpha, beta)
                maxScore = max(maxScore, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
        return maxScore

    else:
        minScore = sys.maxsize
        legal_moves = generate_legal_moves(board, False)

        if len(legal_moves) == 0:
            minScore = minimax_value(board, True, search_depth, alpha, beta)
        
        else:
            for move in legal_moves:
                new_board = play_move(board, move, False) # Get the new board state
                score = minimax_value(new_board, True, search_depth - 1, alpha, beta)
                minScore = min(minScore, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
        return minScore
    
# Printing a board (and return null), for interactive mode
def print_board(board):
    printable = {
        -1: "B",
        0: "-",
        1: "W"
    }
    for r in range(NUM_COLUMNS):
        line = ""
        for c in range(NUM_COLUMNS):
            line += printable[board[r][c]]
        print(line)

# Interactive play, for demo purposes.  Assume AI is white and goes first.
def play():
    board = starting_board()
    while check_game_over(board) == NOBODY:
        # White turn (AI)
        legal_moves = generate_legal_moves(board, True)
        if legal_moves:  # (list is non-empty)
            print("Thinking...")
            best_val = float("-inf")
            best_move = None
            for m in legal_moves:
                new_board = play_move(board, m, True)
                move_val = minimax_value(new_board, True, DEMO_SEARCH_DEPTH, float("-inf"), float("inf"))
                if move_val > best_val:
                    best_move = m
                    best_val = move_val
            board = play_move(board, best_move, True)
            print_board(board)
            print("")
        else:
            print("White has no legal moves; skipping turn...")

        legal_moves = generate_legal_moves(board, False)
        if legal_moves:
            player_move = get_player_move(board, legal_moves)
            board = play_move(board, player_move, False)
            print_board(board)
        else:
            print("Black has no legal moves; skipping turn...")
    winner = find_winner(board)
    if winner == WHITE:
        print("White won!")
    elif winner == BLACK:
        print("Black won!")
    else:
        print("Tie!")

def starting_board():
    board = np.zeros((NUM_COLUMNS, NUM_COLUMNS))
    board[3][3] = WHITE
    board[3][4] = BLACK
    board[4][3] = BLACK
    board[4][4] = WHITE
    return board

# Print board with numbers for the legal move spaces, then get player choice of move
# Returns the (row,col) representation of the choice
def get_player_move(board, legal_moves):
    for r in range(NUM_COLUMNS):
        line = ""
        for c in range(NUM_COLUMNS):
            if board[r][c] == WHITE:
                line += "W"
            elif board[r][c] == BLACK:
                line += "B"
            else:
                if (r,c) in legal_moves:
                    line += str(legal_moves.index((r,c)))
                else:
                    line += "-"
        print(line)
    while True:
        # Bounce around this loop until a valid integer is received
        choice = input("Which move do you want to play? [0-" + str(len(legal_moves)-1) + "]")
        try:
            move_num = int(choice)
            if move_num >= 0 and move_num < len(legal_moves):
                return legal_moves[move_num]
            else:
                print("That wasn't one of the options.")
        except ValueError:
            print("Please enter an integer as your move choice.")


# main
# The input consists of a first line that is the desired search depth, then
# a board description (see read_board).  The desired output is the value of the board
# according to the evaluation function (with minimax recursion).
#
# Alternately, if the line of input is "play" instead, we can launch into an actual
# game for demo purposes.
firstline = input("") # read just one line
if firstline == "play":
    play()
    # sys.exit("Game over")
else:
    try:
        search_depth = int(firstline)
    except ValueError:
        sys.exit("First line was neither 'play' nor a search depth; quitting...")
    board = read_board()
    t0 = time()
    print(minimax_value(board, WHITE_TO_PLAY, search_depth, float("-inf"), float("inf")))
    t1 = time()
    duration = round(t1-t0, 3)
    print(f"Time taken: {duration} seconds")
