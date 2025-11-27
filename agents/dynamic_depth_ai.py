"""
Dynamic Depth AI - Caching AI with Dynamic Search Depth.

This agent adjusts its search depth based on board complexity (empty cells).
Tuned to be less aggressive than the original DynamicAI to avoid timeouts.
"""

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.caching_ai import CachingAI
from game_core import GameState, Direction, Game


class DynamicDepthAI(CachingAI):
    """
    Caching AI with Dynamic Depth (Tuned).
    
    Strategy:
        - Empty > 12: Depth 3 (Fast)
        - Empty > 4: Depth 4 (Standard)
        - Else: Depth 5 (Critical)
    """
    
    def __init__(self, game: Game):
        super().__init__(game, depth=3)
    
    def get_move(self, state: GameState) -> Direction:
        """Choose move using Dynamic Depth."""
        empty_count = state.get_empty_count()
        
        if empty_count > 12:
            search_depth = 3
        elif empty_count > 4:
            search_depth = 4
        else:
            search_depth = 5  # Critical survival mode
            
        self.depth = search_depth
        return super().get_move(state)
