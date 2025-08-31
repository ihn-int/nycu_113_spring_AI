import numpy as np
import random
import game
import math     # For MCTS
import copy     # For MCTS

def print_INFO():
    """
    Prints your homework submission details.
    Please replace the placeholders (date, name, student ID) with valid information
    before submitting.
    """
    print(
        """========================================
        DATE: 2025/04/01
        STUDENT NAME: JUN-KAI CHEN
        STUDENT ID: 111550001
        ========================================
        """)


#
# Basic search functions: Minimax and Alpha‑Beta
#

def minimax(grid, depth, maximizingPlayer, dep=4):
    """
    TODO (Part 1): Implement recursive Minimax search for Connect Four.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
    """

    # Check terminate
    if grid.terminate() or depth == 0:
        return get_heuristic(grid), set()
    
    
    # Check player score
    if maximizingPlayer:
        max_eval = float("-inf")
        best_moves = set()
        # Check only valid columns
        for col in grid.valid:
            new_grid = game.drop_piece(grid, col)
            # Solve the problem by recursive
            eval_score, _ = minimax(new_grid, depth-1, False, dep)
            if eval_score > max_eval:    # better solution
                max_eval = eval_score
                best_moves = { col }
            elif eval_score == max_eval: # as well as current solutions
                best_moves.add(col)
        return max_eval, best_moves
    # Else check opposite score
    else:
        min_eval = float("inf")
        best_moves = set()
        # Check only valid columns
        for col in grid.valid:
            new_grid = game.drop_piece(grid, col)
            # Solve the problem by recursive
            eval_score, _ = minimax(new_grid, depth-1, True, dep)
            if eval_score < min_eval:    # better solution
                min_eval = eval_score
                best_moves = { col }
            elif eval_score == min_eval: # as well as current solutions
                best_moves.add(col)
        return min_eval, best_moves
    
    # Placeholder return to keep function structure intact
    # return 0, {0}


def alphabeta(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    TODO (Part 2): Implement Alpha-Beta pruning as an optimization to Minimax.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
      - Prune branches when alpha >= beta
    """
    # Check terminate
    if grid.terminate() or depth == 0:
        return get_heuristic(grid), set()

    # Check player score
    if maximizingPlayer:
        max_eval = float("-inf")
        best_moves = set()
        # Check omly valid columns
        for col in grid.valid:
            new_grid = game.drop_piece(grid, col)
            eval_score, _ = alphabeta(new_grid, depth-1, False, alpha, beta, dep)
            if eval_score > max_eval:    # better solution
                max_eval = eval_score
                best_moves = { col }
            elif eval_score == max_eval: # as well as current solution
                best_moves.add(col)
            
            # Alpha-Beta pruning
            alpha = max(alpha, eval_score)
            if beta < alpha:
                break
        return max_eval, best_moves
    # Else check opposite score
    else:
        min_eval = float("inf")
        best_moves = set()
        # Check only valid columns
        for col in grid.valid:
            new_grid = game.drop_piece(grid, col)
            eval_score, _ = alphabeta(new_grid,depth-1, True, alpha, beta, dep)
            if eval_score < min_eval:    # better solution
                min_eval = eval_score
                best_moves = { col }
            elif eval_score == min_eval: # as well as current solution
                best_moves.add(col)
            
            # Alpha-Beta pruning
            beta = min(beta, eval_score)
            if beta < alpha:
                break
        return min_eval, best_moves

    # Placeholder return to keep function structure intact
    # return 0, {0}


#
# Basic agents
#

def agent_minimax(grid):
    """
    Agent that uses the minimax() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(minimax(grid, 4, True)[1]))


def agent_alphabeta(grid):
    """
    Agent that uses the alphabeta() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(alphabeta(grid, 4, True, -np.inf, np.inf)[1]))


def agent_reflex(grid):
    """
    A simple reflex agent provided as a baseline:
      - Checks if there's an immediate winning move.
      - Otherwise picks a random valid column.
    """
    wins = [c for c in grid.valid if game.check_winning_move(grid, c, grid.mark)]
    if wins:
        return random.choice(wins)
    return random.choice(grid.valid)


def agent_strong(grid):
    """
    TODO (Part 3): Design your own agent (depth = 4) to consistently beat the Alpha-Beta agent (depth = 4).
    This agent will typically act as Player 2.
    """
    # Placeholder logic that calls your_function().
    return random.choice(list(your_function(grid, 4, False, -np.inf, np.inf)[1]))
    # return mcts_search(grid)


#
# Heuristic functions
#

def get_heuristic(board):
    """
    Evaluates the board from Player 1's perspective using a basic heuristic.

    Returns:
      - Large positive value if Player 1 is winning
      - Large negative value if Player 2 is winning
      - Intermediate scores based on partial connect patterns
    """
    num_twos       = game.count_windows(board, 2, 1)
    num_threes     = game.count_windows(board, 3, 1)
    num_twos_opp   = game.count_windows(board, 2, 2)
    num_threes_opp = game.count_windows(board, 3, 2)

    score = (
          1e10 * board.win(1)
        + 1e6  * num_threes
        + 10   * num_twos
        - 10   * num_twos_opp
        - 1e6  * num_threes_opp
        - 1e10 * board.win(2)
    )
    return score

def get_heuristic_strong(board):
    """
    TODO (Part 3): Implement a more advanced board evaluation for agent_strong.
    Currently a placeholder that returns 0.
    """

    if board.win(1):  # Player 1 獲勝
        return float('inf')
    if board.win(2):  # Player 2 獲勝
        return float('-inf')
    if board.terminate():  # 平手
        return 0

    # 特徵統計
    num_threes     = game.count_windows(board, 3, 1)  # 我方三連
    num_threes_opp = game.count_windows(board, 3, 2)  # 對方三連
    num_twos       = game.count_windows(board, 2, 1)  # 我方二連
    num_twos_opp   = game.count_windows(board, 2, 2)  # 對方二連
    single_counts  = count_single_pieces(board, 1)    # 獨立棋子數

    # 根據特徵給分
    score = (
        900_000 * num_threes - 900_000 * num_threes_opp +
        50_000 * num_twos - 50_000 * num_twos_opp
    )

    # 根據單顆棋子位置給分
    for col, count in single_counts.items():
        if col == 3:  # Column d
            score += count * 200
        elif col in [0, 6]:  # Column a, g
            score += count * 40
        elif col in [1, 5]:  # Column b, f
            score += count * 70
        elif col in [2, 4]:  # Column c, e
            score += count * 120

    return score


def your_function(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    A stronger search function that uses get_heuristic_strong() instead of get_heuristic().
    You can employ advanced features (e.g., improved move ordering, deeper lookahead).

    Return:
      (boardValue, {setOfCandidateMoves})

    Currently a placeholder returning (0, {0}).
    """
    if grid.terminate() or depth == 0:
        return get_heuristic_strong(grid), set()

    best_moves = set()

    if maximizingPlayer:
        max_eval = float('-inf')

        for col in grid.valid:
            new_board = game.drop_piece(grid, col)
            eval_score, _ = your_function(new_board, depth - 1, False, alpha, beta)

            if eval_score > max_eval:
                max_eval = eval_score
                best_moves = {col}
            elif eval_score == max_eval:
                best_moves.add(col)

            alpha = max(alpha, eval_score)
            if beta < alpha:
                break  # 剪枝

        return max_eval, best_moves
    else:
        min_eval = float('inf')

        for col in grid.valid:
            new_board = game.drop_piece(grid, col)
            eval_score, _ = your_function(new_board, depth - 1, True, alpha, beta)

            if eval_score < min_eval:
                min_eval = eval_score
                best_moves = {col}
            elif eval_score == min_eval:
                best_moves.add(col)

            beta = min(beta, eval_score)
            if beta < alpha:
                break  # 剪枝

        return min_eval, best_moves

def count_single_pieces(board, player):
    """
    計算特定玩家在每一列(column)上的 single pieces（沒有相鄰棋子的棋子數量）。
    
    :param board: 當前棋盤狀態
    :param player: 要檢查的玩家（1 或 2）
    :return: 字典 {col_index: single_pieces_count}
    """
    rows, cols = 6, 7  # 取得棋盤大小
    single_pieces = {col: 0 for col in range(cols)}  # 初始化計數

    # 遍歷棋盤上的每個格子
    for r in range(rows):
        for c in range(cols):
            if board.table[r][c] == player:  # 如果這個格子是該玩家的棋子
                # 檢查這個棋子是否是 "single piece"（四周沒有相鄰的同色棋子）
                is_single = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and board.table[nr][nc] == player:
                        is_single = False
                        break  # 只要有相鄰棋子，就不是 single piece
                
                if is_single:
                    single_pieces[c] += 1  # 計入該列的 single piece 數量

    return single_pieces