import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in
                  legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if
                        scores[index] == best_score]
        chosen_index = np.random.choice(
            best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(
            action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile

        "*** YOUR CODE HERE ***"
        score = 0
        if action == Action.UP and board[-1][-1] == max_tile:
            return 0
        elif board[-1][-1] == max_tile:
            score = max_tile

        _merge = np.sum(board[:, :-1][board[:, :-1] == board[:, 1:]]) \
                 + np.sum(board[:-1, :][board[:-1, :] == board[1:, :]])

        return max(_merge, score)


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    _next_move = Action.STOP

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        # util.raiseNotDefined()
        _next_move = Action.STOP
        # self.run_minimax_algorithm(game_state, 0, 0)
        moves = game_state.get_legal_actions(0)
        max_action_move_pairs = np.array([(self.run_minimax_algorithm(
            game_state.generate_successor(0, move), 1,
            1), move) for move in moves], dtype=object)
        max_action, self._next_move = max_action_move_pairs[
            np.argmax(max_action_move_pairs[:, 0])]
        return self._next_move

    def run_minimax_algorithm(self, game_state, index, agent_index):
        if index >= self.depth * 2 or len(game_state.get_legal_actions(agent_index)) == 0:
            self._next_move = Action.STOP
            return self.evaluation_function(game_state)

        moves = game_state.get_legal_actions(agent_index)
        if agent_index == 1:  # Min
            return self.fill_tile(agent_index, game_state, index, moves)

        if agent_index == 0:  # Max
            return self.player_move(agent_index, game_state, index, moves)

    def fill_tile(self, agent_index, game_state, index, moves):
        min_action_move_pairs = np.array([(self.run_minimax_algorithm(
            game_state.generate_successor(agent_index, move), index + 1, 0),
                                           move) for move in moves],
            dtype=object)
        min_action, self._next_move = min_action_move_pairs[
            np.argmin(min_action_move_pairs[:, 0])]
        return min_action

    def player_move(self, agent_index, game_state, index, moves):
        max_action_move_pairs = np.array([(self.run_minimax_algorithm(
            game_state.generate_successor(agent_index, move), index + 1,
            1), move) for move in moves], dtype=object)
        max_action, self._next_move = max_action_move_pairs[
            np.argmax(max_action_move_pairs[:, 0])]
        return max_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    _next_move = Action.STOP

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        self._next_move = Action.STOP
        # self.run_alpha_beta_pruning(game_state, 0, 0, float("-inf"),float("inf"))
        alpha = float("-inf")
        move_vals = []
        for move in game_state.get_legal_actions(0):
            new_alpha = self.run_alpha_beta_pruning(
                game_state.generate_successor(0, move), 1, 1, alpha,
                float("inf"))
            move_vals.append((new_alpha, move))
            if new_alpha > alpha:
                alpha = new_alpha
        move_vals = np.array(move_vals, dtype=object)
        best_idx = np.argmax(move_vals[:, 0])
        alpha, self._next_move = move_vals[best_idx]
        return self._next_move

    def run_alpha_beta_pruning(self, game_state, agent_index, index, alpha,
                               beta):
        if index >= self.depth or len(game_state.get_legal_actions(0)) == 0:
            self._next_move = Action.STOP
            return self.evaluation_function(game_state)

        move_vals = []
        if agent_index == 1:  # Min
            return self.fill_tile(agent_index, alpha, beta, game_state, index,
                                  move_vals)

        if agent_index == 0:  # Max
            return self.player_move(agent_index, alpha, beta, game_state,
                                    index, move_vals)

    def fill_tile(self, agent_index, alpha, beta, game_state, index,
                  move_vals):
        for move in game_state.get_legal_actions(agent_index):
            new_beta = self.run_alpha_beta_pruning(
                game_state.generate_successor(agent_index, move), 0, index,
                alpha, beta)
            move_vals.append((new_beta, move))
            if new_beta < beta:
                beta = new_beta
            if beta <= alpha:
                break
        move_vals = np.array(move_vals, dtype=object)
        best_idx = np.argmin(move_vals[:, 0])
        beta, self._next_move = move_vals[best_idx]
        return beta

    def player_move(self, agent_index, alpha, beta, game_state, index,
                    move_vals):
        for move in game_state.get_legal_actions(agent_index):
            new_alpha = self.run_alpha_beta_pruning(
                game_state.generate_successor(agent_index, move), 1,
                index + 1, alpha, beta)
            move_vals.append((new_alpha, move))
            if new_alpha > alpha:
                alpha = new_alpha
            if beta <= alpha:
                break
        move_vals = np.array(move_vals, dtype=object)
        best_idx = np.argmax(move_vals[:, 0])
        alpha, self._next_move = move_vals[best_idx]
        return alpha


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        opt_action = Action.STOP
        opt_score = float('-inf')
        legal_actions = game_state.get_legal_actions(0)
        for action in legal_actions:
            successor_game_state = game_state.generate_successor(0, action)
            score = self.expectimax(successor_game_state, 1, 1)
            if score > opt_score:
                opt_score = score
                opt_action = action
        return opt_action

    def expectimax(self, game_state, agent_index, depth):
        if depth >= self.depth or len(game_state.get_legal_actions(0)) == 0:
            return self.evaluation_function(game_state)

        if agent_index == 0:
            return self.max_value(game_state, depth)
        else:
            return self.exp_value(game_state, agent_index, depth)

    def max_value(self, game_state, depth):
        max_value = float('-inf')
        for action in game_state.get_legal_actions(0):
            value = self.expectimax(game_state.generate_successor(0, action),
                                    1, depth + 1)
            max_value = max(max_value, value)
        return max_value

    def exp_value(self, game_state, agent_index, depth):
        total_values = 0
        legal_moves = game_state.get_legal_actions(agent_index)
        for action in legal_moves:
            value = self.expectimax(
                game_state.generate_successor(agent_index, action), 0, depth)
            total_values += value / len(legal_moves)
        return total_values


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: This evaluation function assesses the quality of a 2048 game
    state using key heuristics: merges, free tiles, tile value, monotonicity
    and smoothness. Theses were chosen based on gameplay analysis to maximize
    performance, we wrote a simple code, then used numpy to improve the runtime.
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    board = np.array(current_game_state.board)

    # Merges
    merges_score = np.sum(board[:, :-1][board[:, :-1] == board[:, 1:]]) \
                 + np.sum(board[:-1, :][board[:-1, :] == board[1:, :]])

    # Free tiles
    free_tiles_score = np.sum(board == 0)

    # All Values
    all_values_score = np.sum(board)

    # Smoothness
    smoothness_score = -(
            np.sum(np.abs(board[:, :-1] - board[:, 1:])) +
            np.sum(np.abs(board[:-1, :] - board[1:, :]))
    )

    # Monotonicity
    row_diff = np.diff(board, axis=1)
    col_diff = np.diff(board, axis=0)
    row_monotonicity = np.sum((row_diff >= 0) | (row_diff <= 0), axis=1)
    col_monotonicity = np.sum((col_diff >= 0) | (col_diff <= 0), axis=0)
    monotonicity_score = np.sum(
        row_monotonicity == row_diff.shape[1]) + np.sum(
        col_monotonicity == col_diff.shape[0])


    return monotonicity_score + 3 * free_tiles_score + merges_score \
        + 0.2 * all_values_score + 0.1 * smoothness_score



# Abbreviation
better = better_evaluation_function
