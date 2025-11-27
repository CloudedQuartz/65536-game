"""
Simple Heuristic AI - Evaluates each move with a weighted score.
"""

from player_interface import Player
from game_core import GameState, Direction, Game


class HeuristicAI(Player):
    """AI that picks the move with the best heuristic score."""
    
    def __init__(self, game: Game):
        self.game = game
    
    def get_move(self, state: GameState) -> Direction:
        """Choose move with best evaluation."""
        best_move = None
        best_score = float('-inf')
        
        for direction, new_state in self.game.move_engine.get_all_move_outcomes(state).items():
            # Combine multiple heuristics
            score = (
                new_state.get_position_weights() +
                new_state.get_monotonicity() * 1.0 +
                new_state.get_empty_count() * 100.0 -
                new_state.get_smoothness() * 0.1
            )
            
            if score > best_score:
                best_score = score
                best_move = direction
        
        return best_move
    
    def on_move_result(self, result): pass
    def on_game_over(self, state): pass
    def reset(self): pass


if __name__ == "__main__":
    from game_core import SpawnMode
    from main import run_game
    
    game = Game(spawn_mode=SpawnMode.CLASSIC, seed=42)
    player = HeuristicAI(game)
    final = run_game(game, player)
    
    print(f"\nFinal: {final.score:,} points, max tile {final.get_max_tile()}")
