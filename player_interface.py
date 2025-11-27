"""
Player Interface for 65536 Game

This module defines the abstract Player interface that all players
(both human and AI) must implement to interact with the game.

The interface is designed to be simple yet powerful, providing all
necessary information for sophisticated AI while remaining easy to
understand for human player implementations.
"""

from abc import ABC, abstractmethod
from game_core import GameState, Direction, MoveResult


class Player(ABC):
    """
    Abstract base class for all players (human and AI).
    
    This interface defines the contract that all players must follow.
    The game engine calls these methods at specific points during
    the game loop to interact with the player.
    
    Game Loop Flow:
        1. game.reset() -> initial state
        2. Loop while not game_over:
            a. player.get_move(state) -> direction
            b. game.make_move(state, direction) -> (new_state, result)
            c. player.on_move_result(result)
        3. player.on_game_over(final_state)
    
    Contract Guarantees:
        - get_move() will only be called for states where valid moves exist
        - on_move_result() is called after every successful move
        - on_game_over() is called exactly once at the end of the game
        - reset() is called before starting a new game
    
    Implementation Guide:
        To create your own player (AI or human), subclass this class
        and implement all abstract methods. See random_ai_player.py
        for a simple example.
    """
    
    @abstractmethod
    def get_move(self, state: GameState) -> Direction:
        """
        Choose a move for the given game state.
        
        This is the CORE method that defines the player's strategy.
        The game engine calls this method and expects a valid direction.
        
        Args:
            state (GameState): Current game state
                - Access grid with state.grid (4x4 numpy array)
                - Access score with state.score
                - Access move count with state.move_count
                - Use evaluation helpers: state.get_max_tile(), etc.
        
        Returns:
            Direction: One of Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT
        
        Contract:
            - MUST return a valid Direction enum value
            - SHOULD return a direction that changes the board
              (use game.move_engine.get_valid_moves(state) to check)
            - Will only be called when valid moves exist (never in game_over state)
        
        For AI Players:
            You have access to:
            - Current state: state.grid, state.score
            - Valid moves: game.move_engine.get_valid_moves(state)
            - Spawn probabilities: game.tile_spawner.get_spawn_probabilities(state)
            - All successors: game.enumerate_all_successors(state)
            - Evaluation helpers: state.get_smoothness(), state.get_monotonicity(), etc.
        
        Example (Random AI):
            ```python
            def get_move(self, state: GameState) -> Direction:
                valid_moves = self.game.move_engine.get_valid_moves(state)
                return random.choice(valid_moves)
            ```
        
        Example (Heuristic AI):
            ```python
            def get_move(self, state: GameState) -> Direction:
                best_dir = None
                best_score = float('-inf')
                
                for direction in Direction:
                    new_state, changed = self.game.move_engine.execute_move(state, direction)
                    if changed:
                        score = new_state.get_position_weights() - new_state.get_smoothness()
                        if score > best_score:
                            best_score = score
                            best_dir = direction
                
                return best_dir
            ```
        
        Raises:
            Any exception raised will terminate the game
        """
        pass
    
    @abstractmethod
    def on_move_result(self, result: MoveResult) -> None:
        """
        Callback invoked after each successful move.
        
        This method is called immediately after a move is executed and
        a new tile is spawned. Use it for logging, learning, or updating
        internal state.
        
        When Called:
            Called after every successful move, before the next get_move()
        
        Args:
            result (MoveResult): Details about what happened
                - result.success: Was move valid? (always True when called)
                - result.state: New game state after move + spawn
                - result.score_gained: Points earned from merges this move
                - result.merged_tiles: Positions where merges occurred
                - result.spawned_tile: (row, col, value) of newly spawned tile
        
        Returns:
            None
        
        Contract:
            - This method MUST NOT block for long periods
            - Should not raise exceptions (will terminate game)
            - Is optional to use (can be empty pass statement)
        
        Common Uses:
            - Logging: Print move details for debugging
            - Learning: Update neural network weights
            - Statistics: Track move history
            - Validation: Verify AI's predictions matched actual outcomes
        
        Example (Logging):
            ```python
            def on_move_result(self, result: MoveResult) -> None:
                print(f"Move {result.state.move_count}: +{result.score_gained} points")
                if result.spawned_tile:
                    r, c, v = result.spawned_tile
                    print(f"  Spawned {v} at ({r}, {c})")
            ```
        
        Example (Learning):
            ```python
            def on_move_result(self, result: MoveResult) -> None:
                # Update Q-values based on reward
                reward = result.score_gained
                self.update_q_values(self.last_state, self.last_action, reward)
                self.last_state = result.state
            ```
        """
        pass
    
    @abstractmethod
    def on_game_over(self, state: GameState) -> None:
        """
        Callback invoked when the game ends.
        
        This method is called exactly once when no more valid moves remain.
        Use it for final statistics, cleanup, or saving results.
        
        When Called:
            Called once at the end of the game, after the last move
        
        Args:
            state (GameState): Final game state
                - state.game_over: True
                - state.won: True if reached 65536
                - state.score: Final score
                - state.move_count: Total moves made
                - state.get_max_tile(): Highest tile achieved
        
        Returns:
            None
        
        Contract:
            - Called exactly once per game
            - Always called, even if player made zero moves
            - Is optional to use (can be empty pass statement)
        
        Common Uses:
            - Statistics: Print final score and max tile
            - Logging: Save game replay
            - Learning: Save model weights
            - Benchmarking: Record performance metrics
        
        Example (Statistics):
            ```python
            def on_game_over(self, state: GameState) -> None:
                print(f"\\nGame Over!")
                print(f"  Final Score: {state.score}")
                print(f"  Max Tile: {state.get_max_tile()}")
                print(f"  Moves: {state.move_count}")
                if state.won:
                    print(f"  🎉 YOU WON!")
            ```
        
        Example (Save Results):
            ```python
            def on_game_over(self, state: GameState) -> None:
                results = {
                    'score': state.score,
                    'max_tile': state.get_max_tile(),
                    'moves': state.move_count,
                    'won': state.won
                }
                self.game_history.append(results)
            ```
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset player state for a new game.
        
        This method is called before starting a new game. Use it to
        clear any internal state from the previous game.
        
        When Called:
            Called once before each new game starts
        
        Returns:
            None
        
        Contract:
            - Must restore player to initial state
            - Should clear any game-specific internal state
            - Is optional to use (can be empty pass statement)
        
        Common Uses:
            - Clear move history
            - Reset learning parameters
            - Initialize new game statistics
            - Clear cached states
        
        Example (Clear History):
            ```python
            def reset(self) -> None:
                self.move_history = []
                self.last_state = None
                self.last_action = None
            ```
        
        Example (Reset Learning):
            ```python
            def reset(self) -> None:
                self.episode_number += 1
                self.epsilon = max(0.01, self.epsilon * 0.995)  # Decay exploration
                self.episode_rewards = []
            ```
        """
        pass


# ========== Implementation Notes ==========

"""
Implementation Tips for AI Developers:

1. **Access Game Methods**:
   Your AI should keep a reference to the Game instance:
   ```python
   class MyAI(Player):
       def __init__(self, game: Game):
           self.game = game
   ```

2. **Tree Search**:
   Use game.enumerate_all_successors() for Expectimax:
   ```python
   def expectimax(self, state, depth):
       if depth == 0:
           return state.get_position_weights()  # Heuristic
       
       best_value = float('-inf')
       for direction, succ_state, prob in self.game.enumerate_all_successors(state):
           value = prob * self.expectimax(succ_state, depth - 1)
           best_value = max(best_value, value)
       return best_value
   ```

3. **State Evaluation**:
   Combine multiple heuristics:
   ```python
   def evaluate(self, state):
       return (
           state.get_position_weights() * 1.0 +
           -state.get_smoothness() * 0.1 +
           state.get_monotonicity() * 1.0 +
           state.get_merge_potential() * 10.0 +
           state.get_empty_count() * 100.0
       )
   ```

4. **Caching**:
   Use state hashing for transposition tables:
   ```python
   def __init__(self, game):
       self.game = game
       self.cache = {}  # state_hash -> best_move
   
   def get_move(self, state):
       state_hash = hash(state)
       if state_hash in self.cache:
           return self.cache[state_hash]
       # ... compute best move ...
       self.cache[state_hash] = best_move
       return best_move
   ```

5. **Time Management**:
   For real-time play, limit search depth:
   ```python
   import time
   
   def get_move(self, state):
       start_time = time.time()
       max_time = 1.0  # 1 second per move
       
       depth = 1
       best_move = None
       
       while time.time() - start_time < max_time:
           move = self.search(state, depth)
           if move:
               best_move = move
           depth += 1
       
       return best_move
   ```
"""
