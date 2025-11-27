"""
Caching AI - Stochastic Gradient AI with Transposition Table.

This agent adds a transposition table (cache) to the Stochastic Gradient AI.
It stores evaluated states to avoid re-calculating the same positions,
which significantly speeds up the search.
"""

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from player_interface import Player
from game_core import GameState, Direction, Game
import evaluator


class CachingAI(Player):
    """
    Stochastic Gradient AI with Transposition Table.
    
    Features:
        - Gradient Heuristic (Snake Pattern)
        - Stochastic Pruning
        - Transposition Table (Cache)
    """
    
    def __init__(self, game: Game, depth: int = 4, prob_threshold: float = 0.001):
        self.game = game
        self.depth = depth
        self.prob_threshold = prob_threshold
        
        # Cache: state_hash -> score
        self.transposition_table = {}
        
        # Optimized weights (Gradient + Survival)
        self.weights = {
            'gradient': 1.0,
            'empty_cells': 100.0,
            'smoothness': -0.1
        }
    
    def get_move(self, state: GameState) -> Direction:
        """Choose move using Cached Pruned Expectimax."""
        # Clear cache periodically to prevent memory explosion
        if len(self.transposition_table) > 100000:
            self.transposition_table.clear()
            
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
        """Compute expected value with caching."""
        # Check cache (key: state + depth)
        # We include depth because a deeper search is more accurate
        cache_key = (state, depth)
        if cache_key in self.transposition_table:
            return self.transposition_table[cache_key]
        
        if depth == 0 or self.game.is_game_over(state):
            score = evaluator.evaluate_state(state, self.weights)
            self.transposition_table[cache_key] = score
            return score
        
        if prob < self.prob_threshold:
            score = evaluator.evaluate_state(state, self.weights)
            self.transposition_table[cache_key] = score
            return score
        
        expected = 0.0
        possible_spawns = self.game.tile_spawner.get_all_possible_spawns(state)
        
        if not possible_spawns:
            return evaluator.evaluate_state(state, self.weights)
            
        for spawn_state, spawn_prob in possible_spawns:
            new_prob = prob * spawn_prob
            value = self._max_value(spawn_state, depth - 1, new_prob)
            expected += spawn_prob * value
        
        self.transposition_table[cache_key] = expected
        return expected
    
    def _max_value(self, state: GameState, depth: int, prob: float) -> float:
        """Maximize player move."""
        cache_key = (state, depth)
        if cache_key in self.transposition_table:
            return self.transposition_table[cache_key]
            
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
        
        self.transposition_table[cache_key] = best
        return best
    
    def on_move_result(self, result): pass
    def on_game_over(self, state): pass
    def reset(self): 
        self.transposition_table.clear()
