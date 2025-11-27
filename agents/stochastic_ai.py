"""
Stochastic AI - Expectimax with Probability-Based Pruning.

This agent uses the standard heuristics (Position + Monotonicity + Smoothness)
but adds stochastic pruning to search deeper.
"""

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from player_interface import Player
from game_core import GameState, Direction, Game
import evaluator


class StochasticAI(Player):
    """
    Expectimax AI with Stochastic Pruning.
    
    Pruning Strategy:
        - Stop searching if cumulative probability drops below threshold
        - Focuses computation on likely outcomes
        - Allows deeper search depth than standard Expectimax
    """
    
    def __init__(self, game: Game, depth: int = 3, prob_threshold: float = 0.001):
        """
        Initialize agent.
        
        Args:
            game: Game instance
            depth: Search depth (default: 3)
            prob_threshold: Pruning threshold (0.0 to 1.0)
                           Branches with prob < threshold are pruned
        """
        self.game = game
        self.depth = depth
        self.prob_threshold = prob_threshold
        
        # Standard weights (Position + Monotonicity + Smoothness)
        self.weights = {
            'position': 1.0,
            'monotonicity': 1.0,
            'smoothness': -0.1,
            'empty_cells': 100.0,
            'merge_potential': 10.0
        }
    
    def get_move(self, state: GameState) -> Direction:
        """Choose move using Pruned Expectimax."""
        best_move = None
        best_value = float('-inf')
        
        # Get all valid moves
        moves = self.game.move_engine.get_all_move_outcomes(state)
        if not moves:
            return Direction.UP  # Game over or no moves
            
        for direction, after_move in moves.items():
            # Start search with probability 1.0
            value = self._expect_value(after_move, self.depth - 1, 1.0)
            if value > best_value:
                best_value = value
                best_move = direction
        
        return best_move
    
    def _expect_value(self, state: GameState, depth: int, prob: float) -> float:
        """
        Compute expected value with pruning.
        
        Args:
            state: Current state
            depth: Remaining depth
            prob: Cumulative probability of reaching this node
        """
        # Base case: Depth 0 or Game Over
        if depth == 0 or self.game.is_game_over(state):
            return evaluator.evaluate_state(state, self.weights)
        
        # Pruning: If probability is too low, stop searching and evaluate
        if prob < self.prob_threshold:
            return evaluator.evaluate_state(state, self.weights)
        
        expected = 0.0
        possible_spawns = self.game.tile_spawner.get_all_possible_spawns(state)
        
        if not possible_spawns:
            return evaluator.evaluate_state(state, self.weights)
            
        for spawn_state, spawn_prob in possible_spawns:
            new_prob = prob * spawn_prob
            value = self._max_value(spawn_state, depth - 1, new_prob)
            expected += spawn_prob * value
        
        return expected
    
    def _max_value(self, state: GameState, depth: int, prob: float) -> float:
        """Maximize player move."""
        # Base case
        if depth == 0 or self.game.is_game_over(state):
            return evaluator.evaluate_state(state, self.weights)
        
        # Pruning check
        if prob < self.prob_threshold:
            return evaluator.evaluate_state(state, self.weights)
        
        best = float('-inf')
        moves = self.game.move_engine.get_all_move_outcomes(state)
        
        if not moves:
            return evaluator.evaluate_state(state, self.weights)
            
        for direction, after_move in moves.items():
            # Probability doesn't change on player move
            value = self._expect_value(after_move, depth - 1, prob)
            best = max(best, value)
        
        return best
    
    def on_move_result(self, result): pass
    def on_game_over(self, state): pass
    def reset(self): pass
