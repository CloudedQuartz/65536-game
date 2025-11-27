"""
Dynamic Pruning AI - Caching AI with Dynamic Pruning Threshold.

This agent adjusts its risk tolerance based on board complexity.
It becomes extremely careful (low pruning threshold) when the board is full.
"""

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.caching_ai import CachingAI
from game_core import GameState, Direction, Game


class DynamicPruningAI(CachingAI):
    """
    Caching AI with Dynamic Pruning.
    
    Strategy:
        - Empty > 4: Threshold 0.001 (Fast, aggressive pruning)
        - Empty <= 4: Threshold 0.00001 (Careful, minimal pruning)
    """
    
    def __init__(self, game: Game):
        super().__init__(game, depth=4)
    
    def get_move(self, state: GameState) -> Direction:
        """Choose move using Dynamic Pruning."""
        empty_count = state.get_empty_count()
        
        if empty_count > 4:
            self.prob_threshold = 0.001
        else:
            self.prob_threshold = 0.00001  # Be very careful
            
        return super().get_move(state)
