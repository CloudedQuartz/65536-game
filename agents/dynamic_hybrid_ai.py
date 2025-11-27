"""
Dynamic Hybrid AI - Caching AI with Dynamic Depth AND Pruning.

This agent combines dynamic depth and dynamic pruning for maximum performance.
It starts fast and loose, and becomes deep and careful in the endgame.
"""

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.caching_ai import CachingAI
from game_core import GameState, Direction, Game


class DynamicHybridAI(CachingAI):
    """
    Caching AI with Dynamic Depth & Pruning.
    
    Strategy:
        - Empty > 12: Depth 3, Threshold 0.001 (Fastest)
        - Empty > 4: Depth 4, Threshold 0.0001 (Balanced)
        - Else: Depth 5, Threshold 0.00001 (Deep & Careful)
    """
    
    def __init__(self, game: Game):
        super().__init__(game, depth=3)
    
    def get_move(self, state: GameState) -> Direction:
        """Choose move using Dynamic Hybrid Strategy."""
        empty_count = state.get_empty_count()
        
        if empty_count > 12:
            self.depth = 3
            self.prob_threshold = 0.001
        elif empty_count > 4:
            self.depth = 4
            self.prob_threshold = 0.0001
        else:
            self.depth = 5
            self.prob_threshold = 0.00001  # Critical survival mode
            
        return super().get_move(state)
