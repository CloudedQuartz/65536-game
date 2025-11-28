"""
Simple Heuristic AI - Evaluates each move with a weighted score.
"""

import sys
from pathlib import Path
# Add parent directory to path so we can import from root
sys.path.insert(0, str(Path(__file__).parent.parent))

from player_interface import Player
from game_core import GameState, Direction, Game
import evaluator


class HeuristicAI(Player):
    """AI that picks the move with the best heuristic score."""
    
    def __init__(self, game: Game):
        self.game = game
    
    def get_move(self, state: GameState) -> Direction:
        """Choose move with best evaluation."""
        best_move = None
        best_score = float('-inf')
        
        for direction, new_state in self.game.move_engine.get_all_move_outcomes(state).items():
            # Combine multiple heuristics using evaluator module
            # Note: get_gradient_score is scaled by 2.0 to match old position weights
            score = (
                evaluator.get_gradient_score(new_state) * 2.0 +
                evaluator.get_monotonicity(new_state) * 1.0 +
                new_state.get_empty_count() * 100.0 -
                evaluator.get_smoothness(new_state) * 0.1
            )
            
            if score > best_score:
                best_score = score
                best_move = direction
        
        return best_move
    
    def on_move_result(self, result): pass
    def on_game_over(self, state): pass
    def reset(self): pass
