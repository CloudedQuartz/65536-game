"""
Stochastic Gradient AI - Pruned Expectimax with Gradient Heuristic.

This agent uses the Gradient (Snake) heuristic which forces a monotonic chain
without complex logic, combined with stochastic pruning for deeper search.
"""

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from player_interface import Player
from game_core import GameState, Direction, Game
import evaluator


class StochasticGradientAI(Player):
    """
    Expectimax AI with Stochastic Pruning and Gradient Heuristic.
    
    Features:
        - Gradient Heuristic: Forces optimal 'snake' pattern
        - Stochastic Pruning: Ignores unlikely branches
        - High Performance: Fast evaluation + efficient search
    """
    
    def __init__(self, game: Game, depth: int = 3, prob_threshold: float = 0.001):
        """
        Initialize agent.
        
        Args:
            game: Game instance
            depth: Search depth (default: 3)
            prob_threshold: Pruning threshold (0.0 to 1.0)
        """
        self.game = game
        self.depth = depth
        self.prob_threshold = prob_threshold
        
        # Gradient weights (Snake Pattern + Survival)
        self.weights = {
            'gradient': 1.0,      # Strong snake pattern
            'empty_cells': 100.0, # Survival
            'smoothness': -0.1,   # Local mergeability
        }
    
    def get_move(self, state: GameState) -> Direction:
        """Choose move using Pruned Expectimax."""
        best_move = None
        best_value = float('-inf')
        
        moves = self.game.move_engine.get_all_move_outcomes(state)
        if not moves:
            return Direction.UP
            
        for direction, after_move in moves.items():
            value = self._expect_value(after_move, self.depth - 1, 1.0)
            if value > best_value:
                best_value = value
                best_move = direction
        
        return best_move
    
    def _expect_value(self, state: GameState, depth: int, prob: float) -> float:
        """Compute expected value with pruning."""
        if depth == 0 or self.game.is_game_over(state):
            return evaluator.evaluate_state(state, self.weights)
        
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
        if depth == 0 or self.game.is_game_over(state):
            return evaluator.evaluate_state(state, self.weights)
        
        if prob < self.prob_threshold:
            return evaluator.evaluate_state(state, self.weights)
        
        best = float('-inf')
        moves = self.game.move_engine.get_all_move_outcomes(state)
        
        if not moves:
            return evaluator.evaluate_state(state, self.weights)
            
        for direction, after_move in moves.items():
            value = self._expect_value(after_move, depth - 1, prob)
            best = max(best, value)
        
        return best
    
    def on_move_result(self, result): pass
    def on_game_over(self, state): pass
    def reset(self): pass
