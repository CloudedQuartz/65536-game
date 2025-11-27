"""
Simple Expectimax AI - Tree search with depth 2.
"""

from player_interface import Player
from game_core import GameState, Direction, Game


class ExpectimaxAI(Player):
    """AI using Expectimax algorithm (depth 2 recommended)."""
    
    def __init__(self, game: Game, depth: int = 2):
        self.game = game
        self.depth = depth
    
    def get_move(self, state: GameState) -> Direction:
        """Choose move using Expectimax search."""
        best_move = None
        best_value = float('-inf')
        
        for direction, after_move in self.game.move_engine.get_all_move_outcomes(state).items():
            value = self._expect_value(after_move, self.depth - 1)
            if value > best_value:
                best_value = value
                best_move = direction
        
        return best_move
    
    def _expect_value(self, state: GameState, depth: int) -> float:
        """Compute expected value over tile spawns."""
        if depth == 0 or self.game.is_game_over(state):
            return self._evaluate(state)
        
        expected = 0.0
        for spawn_state, prob in self.game.tile_spawner.get_all_possible_spawns(state):
            value = self._max_value(spawn_state, depth - 1)
            expected += prob * value
        
        return expected
    
    def _max_value(self, state: GameState, depth: int) -> float:
        """Choose move that maximizes value."""
        if depth == 0 or self.game.is_game_over(state):
            return self._evaluate(state)
        
        best = float('-inf')
        for direction, after_move in self.game.move_engine.get_all_move_outcomes(state).items():
            value = self._expect_value(after_move, depth - 1)
            best = max(best, value)
        
        return best
    
    def _evaluate(self, state: GameState) -> float:
        """Heuristic evaluation."""
        return (
            state.get_position_weights() +
            state.get_monotonicity() * 1.0 +
            state.get_empty_count() * 100.0 -
            state.get_smoothness() * 0.1
        )
    
    def on_move_result(self, result): pass
    def on_game_over(self, state): pass
    def reset(self): pass


if __name__ == "__main__":
    from game_core import SpawnMode
    from main import run_game
    
    game = Game(spawn_mode=SpawnMode.CLASSIC, seed=42)
    player = ExpectimaxAI(game, depth=2)
    final = run_game(game, player)
    
    print(f"\nFinal: {final.score:,} points, max tile {final.get_max_tile()}")
