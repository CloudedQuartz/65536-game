"""
65536 Game Core Logic

This module contains the core game logic for the 65536 game (similar to 2048).
It is designed to be completely independent of any UI or player implementation,
with a focus on AI-readiness and efficient tree search.

Key Design Principles:
- Immutable game states for easy tree search
- Exposed probabilities for Expectimax algorithms
- Efficient state hashing and enumeration
- Support for both Classic (2048-style) and Dynamic (65536 1/16th rule) spawning
"""

from __future__ import annotations
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import struct


class Direction(Enum):
    """
    Represents the four possible move directions.
    
    Used by both players and the game engine to specify moves.
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SpawnMode(Enum):
    """
    Tile spawning modes supported by the game.
    
    CLASSIC: 2048-style spawning
        - Always spawns 2 (90%) or 4 (10%)
        - Fixed branching factor for AI search
    
    DYNAMIC: 65536-style spawning with 1/16th rule
        - Max spawn value = highest_tile / 16
        - Uniform distribution among valid powers of 2
        - Growing branching factor as game progresses
    """
    CLASSIC = "classic"
    DYNAMIC = "dynamic"


class GameState:
    """
    Represents the complete state of a 65536 game at any point in time.
    
    This class is designed to be immutable for efficient tree search. The grid
    is stored as a NumPy array but should be treated as read-only.
    
    Attributes:
        grid (np.ndarray): 4x4 array of integers, 0 = empty cell
        score (int): Current score
        move_count (int): Number of moves made so far
        game_over (bool): Whether the game has ended
        won (bool): Whether player has reached 65536
        
    Performance Notes:
        - Hashing is cached for O(1) lookups in transposition tables
        - Grid is stored as immutable view to enable shallow copying
        - Equality check is optimized with hash comparison first
    """
    
    def __init__(self, grid: np.ndarray, score: int = 0, move_count: int = 0,
                 game_over: bool = False, won: bool = False):
        """
        Initialize a game state.
        
        Args:
            grid: 4x4 numpy array of tile values (0 = empty)
            score: Current score
            move_count: Number of moves made
            game_over: Whether game has ended
            won: Whether 65536 tile has been reached
        """
        self.grid = grid.copy()
        self.grid.flags.writeable = False  # Make immutable
        self.score = score
        self.move_count = move_count
        self.game_over = game_over
        self.won = won
        self._hash: Optional[int] = None
    
    def copy(self) -> GameState:
        """
        Create a copy of this state.
        
        This is an efficient shallow copy since the grid is immutable.
        
        Returns:
            New GameState with same values
        """
        new_grid = self.grid.copy()
        return GameState(new_grid, self.score, self.move_count, 
                        self.game_over, self.won)
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """
        Get list of empty cell positions.
        
        Returns:
            List of (row, col) tuples for cells with value 0
        """
        return [(i, j) for i in range(4) for j in range(4) if self.grid[i, j] == 0]
    
    def get_empty_count(self) -> int:
        """
        Fast count of empty cells.
        
        Returns:
            Number of cells with value 0
        """
        return int(np.sum(self.grid == 0))
    
    def is_move_valid(self, direction: Direction) -> bool:
        """
        Check if a move in the given direction would change the board.
        
        This is a lightweight check that doesn't actually execute the move.
        
        Args:
            direction: Direction to check
            
        Returns:
            True if move would change the board, False otherwise
        """
        # This is implemented by attempting the move and checking if anything changed
        # A more optimized version could check without full execution
        from copy import deepcopy
        test_grid = self.grid.copy()
        test_grid.flags.writeable = True
        
        engine = MoveEngine()
        new_state, changed = engine.execute_move(
            GameState(test_grid, self.score, self.move_count), 
            direction
        )
        return changed
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality with another GameState.
        
        Two states are equal if they have the same grid configuration and score.
        Move count is not considered for equality (for transposition tables).
        
        Args:
            other: Object to compare with
            
        Returns:
            True if states are equal, False otherwise
        """
        if not isinstance(other, GameState):
            return False
        
        # Quick hash check first
        if self._hash is not None and other._hash is not None:
            if self._hash != other._hash:
                return False
        
        return (np.array_equal(self.grid, other.grid) and 
                self.score == other.score)
    
    def __hash__(self) -> int:
        """
        Compute hash for this state.
        
        Hash is based on grid configuration and score. Cached for performance.
        
        Algorithm:
            - Convert grid to bytes
            - Combine with score using XOR
            - Cache result
            
        Returns:
            Integer hash value
        """
        if self._hash is None:
            # Convert grid to bytes and hash
            grid_bytes = self.grid.tobytes()
            grid_hash = hash(grid_bytes)
            # Combine with score
            self._hash = grid_hash ^ hash(self.score)
        return self._hash
    
    def to_tuple(self) -> Tuple:
        """
        Convert state to hashable tuple.
        
        Useful for using states as dictionary keys.
        
        Returns:
            Tuple of (grid_tuple, score)
        """
        return (tuple(self.grid.flatten()), self.score)
    
    def serialize(self) -> bytes:
        """
        Serialize state to compact binary format.
        
        Format:
            - 64 bytes: grid (16 values × 4 bytes each as uint32)
            - 4 bytes: score (uint32)
            - 4 bytes: move_count (uint32)
            - 1 byte: flags (game_over, won)
            
        Returns:
            Binary representation of state
        """
        data = bytearray()
        
        # Grid: 16 values as uint32
        for val in self.grid.flatten():
            data.extend(struct.pack('I', int(val)))
        
        # Score and move_count
        data.extend(struct.pack('I', self.score))
        data.extend(struct.pack('I', self.move_count))
        
        # Flags
        flags = (int(self.game_over) << 1) | int(self.won)
        data.extend(struct.pack('B', flags))
        
        return bytes(data)
    
    @staticmethod
    def deserialize(data: bytes) -> GameState:
        """
        Restore state from serialized binary format.
        
        Args:
            data: Binary data from serialize()
            
        Returns:
            Reconstructed GameState
        """
        # Grid
        grid_values = []
        for i in range(16):
            val = struct.unpack('I', data[i*4:(i+1)*4])[0]
            grid_values.append(val)
        grid = np.array(grid_values, dtype=np.int32).reshape(4, 4)
        
        # Score and move_count
        score = struct.unpack('I', data[64:68])[0]
        move_count = struct.unpack('I', data[68:72])[0]
        
        # Flags
        flags = struct.unpack('B', data[72:73])[0]
        game_over = bool((flags >> 1) & 1)
        won = bool(flags & 1)
        
        return GameState(grid, score, move_count, game_over, won)
    
    
    def get_max_tile(self) -> int:
        """
        Get the highest tile value on the board.
        
        Returns:
            Maximum tile value (0 if board is empty)
            
        Complexity: O(1) - uses NumPy max
        
        Example:
            Grid with tiles [2, 4, 8, 16] returns 16
        """
        return int(np.max(self.grid))



class TileSpawner:
    """
    Handles tile spawning with support for multiple spawning modes.
    
    Supports:
        - Classic mode: 90% chance of 2, 10% chance of 4
        - Dynamic mode: 1/16th rule with uniform distribution
    
    All probabilities are exposed for AI algorithms like Expectimax.
    """
    
    def __init__(self, mode: SpawnMode = SpawnMode.CLASSIC):
        """
        Initialize tile spawner.
        
        Args:
            mode: Spawning mode to use
        """
        self.mode = mode
        self._cache: Dict[int, Dict[int, float]] = {}
    
    def get_spawn_probabilities(self, state: GameState) -> Dict[int, float]:
        """
        Get probability distribution for tile spawning.
        
        Args:
            state: Current game state
            
        Returns:
            Dictionary mapping tile value to probability
            
        Classic Mode:
            Always returns {2: 0.9, 4: 0.1}
            
        Dynamic Mode:
            Returns uniform distribution from 2 to max_tile // 16
            Example: If max_tile = 1024, returns:
                {2: 1/6, 4: 1/6, 8: 1/6, 16: 1/6, 32: 1/6, 64: 1/6}
        """
        if self.mode == SpawnMode.CLASSIC:
            return {2: 0.9, 4: 0.1}
        
        # Dynamic mode
        max_tile = state.get_max_tile()
        
        # Cache check
        if max_tile in self._cache:
            return self._cache[max_tile]
        
        # Calculate valid spawn values (powers of 2 from 2 to max_tile // 16)
        max_spawn = max(2, max_tile // 16)
        
        valid_values = []
        value = 2
        while value <= max_spawn:
            valid_values.append(value)
            value *= 2
        
        # Uniform distribution
        prob = 1.0 / len(valid_values)
        result = {val: prob for val in valid_values}
        
        # Cache for performance
        self._cache[max_tile] = result
        
        return result
    
    def get_valid_spawn_values(self, state: GameState) -> List[int]:
        """
        Get list of possible spawn values for current state.
        
        Args:
            state: Current game state
            
        Returns:
            List of possible tile values that can spawn
        """
        return list(self.get_spawn_probabilities(state).keys())
    
    def spawn_tile(self, state: GameState, rng: np.random.Generator) -> GameState:
        """
        Add a random tile to the board.
        
        Args:
            state: Current game state
            rng: NumPy random generator
            
        Returns:
            New state with spawned tile
            
        Raises:
            ValueError: If no empty cells available
        """
        empty_cells = state.get_empty_cells()
        if not empty_cells:
            raise ValueError("No empty cells to spawn tile")
        
        # Choose random position
        pos = empty_cells[rng.integers(0, len(empty_cells))]
        
        # Choose value based on probabilities
        probs = self.get_spawn_probabilities(state)
        values = list(probs.keys())
        probabilities = list(probs.values())
        
        value = rng.choice(values, p=probabilities)
        
        return self.spawn_tile_at(state, pos[0], pos[1], value)
    
    def spawn_tile_at(self, state: GameState, row: int, col: int, 
                      value: int) -> GameState:
        """
        Spawn a tile at a specific position (deterministic).
        
        Useful for testing and AI tree search.
        
        Args:
            state: Current game state
            row: Row index (0-3)
            col: Column index (0-3)
            value: Tile value to spawn
            
        Returns:
            New state with spawned tile
            
        Raises:
            ValueError: If position is not empty
        """
        if state.grid[row, col] != 0:
            raise ValueError(f"Position ({row}, {col}) is not empty")
        
        new_state = state.copy()
        new_state.grid.flags.writeable = True
        new_state.grid[row, col] = value
        new_state.grid.flags.writeable = False
        
        return new_state
    
    def get_all_possible_spawns(self, state: GameState) -> List[Tuple[GameState, float]]:
        """
        Get all possible states after spawning a tile.
        
        This is CRITICAL for Expectimax algorithm implementation.
        
        Args:
            state: Current game state (after a move, before spawn)
            
        Returns:
            List of (new_state, probability) tuples for all possible spawns
            
        Example:
            If 3 empty cells and classic mode:
                Returns 6 states (3 positions × 2 values)
                Each state has probability = P(value) × P(position)
                
        Complexity:
            Classic mode: O(empty_cells × 2)
            Dynamic mode: O(empty_cells × log(max_tile))
        """
        empty_cells = state.get_empty_cells()
        if not empty_cells:
            return []
        
        probs = self.get_spawn_probabilities(state)
        num_empty = len(empty_cells)
        position_prob = 1.0 / num_empty
        
        results = []
        
        for row, col in empty_cells:
            for value, value_prob in probs.items():
                new_state = self.spawn_tile_at(state, row, col, value)
                combined_prob = value_prob * position_prob
                results.append((new_state, combined_prob))
        
        return results


class MoveEngine:
    """
    Handles move execution and board transformations.
    
    Implements the core merging logic for the 2048/65536 game.
    """
    
    def __init__(self):
        """Initialize move engine."""
        pass
    
    def _merge_line(self, line: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Merge a single line (row or column) to the left.
        
        Algorithm:
            1. Remove zeros (compact non-zero tiles to left)
            2. Merge adjacent equal tiles from left to right
            3. Each tile can only merge once per move
            4. Pad with zeros on the right
            
        Args:
            line: 1D array of 4 values
            
        Returns:
            Tuple of (merged_line, score_gained)
            
        Example:
            [2, 2, 4, 0] -> [4, 4, 0, 0], score = 4
            [2, 2, 2, 2] -> [4, 4, 0, 0], score = 8
            [2, 0, 2, 4] -> [4, 4, 0, 0], score = 4
            
        Complexity: O(1) - fixed size 4
        """
        # Remove zeros
        non_zero = line[line != 0]
        
        if len(non_zero) == 0:
            return np.zeros(4, dtype=np.int32), 0
        
        # Merge adjacent equal tiles
        merged = []
        score = 0
        i = 0
        
        while i < len(non_zero):
            if i < len(non_zero) - 1 and non_zero[i] == non_zero[i + 1]:
                # Merge
                new_value = non_zero[i] * 2
                merged.append(new_value)
                score += new_value
                i += 2  # Skip both tiles
            else:
                # No merge
                merged.append(non_zero[i])
                i += 1
        
        # Pad with zeros
        result = np.array(merged + [0] * (4 - len(merged)), dtype=np.int32)
        return result, score
    
    def execute_move(self, state: GameState, direction: Direction) -> Tuple[GameState, bool]:
        """
        Execute a move in the given direction.
        
        Args:
            state: Current game state
            direction: Direction to move
            
        Returns:
            Tuple of (new_state, changed)
                new_state: State after move
                changed: True if board was modified, False if move was illegal
                
        Algorithm:
            1. Rotate board so move is always "left"
            2. Merge each row to the left
            3. Rotate back
            4. Check if anything changed
        """
        new_state = state.copy()
        new_state.grid.flags.writeable = True
        
        # Store original for comparison
        original = state.grid.copy()
        
        total_score = 0
        
        if direction == Direction.LEFT:
            for i in range(4):
                new_state.grid[i, :], score = self._merge_line(new_state.grid[i, :])
                total_score += score
                
        elif direction == Direction.RIGHT:
            for i in range(4):
                # Flip, merge, flip back
                flipped = np.flip(new_state.grid[i, :])
                merged, score = self._merge_line(flipped)
                new_state.grid[i, :] = np.flip(merged)
                total_score += score
                
        elif direction == Direction.UP:
            for j in range(4):
                new_state.grid[:, j], score = self._merge_line(new_state.grid[:, j])
                total_score += score
                
        elif direction == Direction.DOWN:
            for j in range(4):
                # Flip, merge, flip back
                flipped = np.flip(new_state.grid[:, j])
                merged, score = self._merge_line(flipped)
                new_state.grid[:, j] = np.flip(merged)
                total_score += score
        
        # Check if anything changed
        changed = not np.array_equal(original, new_state.grid)
        
        if changed:
            new_state.score += total_score
            new_state.move_count += 1
        
        new_state.grid.flags.writeable = False
        
        return new_state, changed
    
    def get_valid_moves(self, state: GameState) -> List[Direction]:
        """
        Get list of valid moves (moves that would change the board).
        
        Args:
            state: Current game state
            
        Returns:
            List of valid directions
            
        Optimization:
            This checks all 4 directions without full move execution
            where possible, but currently uses execute_move for simplicity.
        """
        valid = []
        
        for direction in Direction:
            _, changed = self.execute_move(state, direction)
            if changed:
                valid.append(direction)
        
        return valid
    
    def get_all_move_outcomes(self, state: GameState) -> Dict[Direction, GameState]:
        """
        Get all possible states after valid moves (before tile spawn).
        
        This is KEY for AI tree search - it returns the deterministic
        move outcomes, before the stochastic tile spawn.
        
        Args:
            state: Current game state
            
        Returns:
            Dictionary mapping each valid direction to resulting state
            Only includes moves that change the board
            
        Usage:
            For Minimax-style search where you want to separate
            the deterministic move phase from random spawn phase.
        """
        outcomes = {}
        
        for direction in Direction:
            new_state, changed = self.execute_move(state, direction)
            if changed:
                outcomes[direction] = new_state
        
        return outcomes


@dataclass
class MoveResult:
    """
    Details about the outcome of a move.
    
    Provides feedback to players about what happened during the move.
    
    Attributes:
        success: Was the move legal (did anything change)?
        state: New game state after move
        score_gained: Points earned from merges this move
        merged_tiles: List of positions where merges occurred
        spawned_tile: (row, col, value) of spawned tile, or None if move failed
    """
    success: bool
    state: GameState
    score_gained: int
    merged_tiles: List[Tuple[int, int]]
    spawned_tile: Optional[Tuple[int, int, int]]


class Game:
    """
    Main game controller that ties all components together.
    
    Manages game initialization, move execution, and provides helpers
    for AI tree search algorithms.
    """
    
    def __init__(self, spawn_mode: SpawnMode = SpawnMode.CLASSIC, 
                 seed: Optional[int] = None):
        """
        Initialize a new game.
        
        Args:
            spawn_mode: Choose between CLASSIC (2048-style) or DYNAMIC (65536-style)
            seed: Optional random seed for reproducibility
        """
        self.spawn_mode = spawn_mode
        self.tile_spawner = TileSpawner(spawn_mode)
        self.move_engine = MoveEngine()
        self.rng = np.random.default_rng(seed)
    
    def reset(self) -> GameState:
        """
        Start a new game.
        
        Returns:
            Initial game state with 2 tiles (both value 2)
        """
        # Create empty grid
        grid = np.zeros((4, 4), dtype=np.int32)
        state = GameState(grid)
        
        # Spawn 2 initial tiles (always value 2)
        pos1 = self.rng.integers(0, 16)
        row1, col1 = pos1 // 4, pos1 % 4
        state = self.tile_spawner.spawn_tile_at(state, row1, col1, 2)
        
        # Spawn second tile at different position
        empty_cells = state.get_empty_cells()
        pos2 = empty_cells[self.rng.integers(0, len(empty_cells))]
        state = self.tile_spawner.spawn_tile_at(state, pos2[0], pos2[1], 2)
        
        return state
    
    def make_move(self, state: GameState, direction: Direction) -> Tuple[GameState, MoveResult]:
        """
        Execute a move and spawn a new tile.
        
        Args:
            state: Current game state
            direction: Direction to move
            
        Returns:
            Tuple of (new_state, move_result)
            
        Process:
            1. Execute move in given direction
            2. If move was valid, spawn new tile
            3. Check for win/loss conditions
            4. Return new state and details
        """
        # Execute move
        new_state, changed = self.move_engine.execute_move(state, direction)
        
        if not changed:
            # Invalid move
            return state, MoveResult(
                success=False,
                state=state,
                score_gained=0,
                merged_tiles=[],
                spawned_tile=None
            )
        
        score_gained = new_state.score - state.score
        
        # Spawn new tile
        try:
            spawned_state = self.tile_spawner.spawn_tile(new_state, self.rng)
            
            # Find where tile was spawned (compare grids)
            spawned_pos = None
            for i in range(4):
                for j in range(4):
                    if new_state.grid[i, j] != spawned_state.grid[i, j]:
                        spawned_pos = (i, j, int(spawned_state.grid[i, j]))
                        break
                if spawned_pos:
                    break
        except ValueError:
            # No empty cells - game over
            spawned_state = new_state
            spawned_state.game_over = True
            spawned_pos = None
        
        # Check win condition
        if spawned_state.get_max_tile() >= 65536:
            spawned_state.won = True
        
        # Check game over (no valid moves)
        if not spawned_state.game_over:
            valid_moves = self.move_engine.get_valid_moves(spawned_state)
            if not valid_moves:
                spawned_state.game_over = True
        
        return spawned_state, MoveResult(
            success=True,
            state=spawned_state,
            score_gained=score_gained,
            merged_tiles=[],  # TODO: Track merge positions if needed
            spawned_tile=spawned_pos
        )
    
    def is_game_over(self, state: GameState) -> bool:
        """
        Check if game is over (no valid moves remain).
        
        Args:
            state: Game state to check
            
        Returns:
            True if no valid moves, False otherwise
        """
        if state.game_over:
            return True
        
        valid_moves = self.move_engine.get_valid_moves(state)
        return len(valid_moves) == 0
    
    # ========== AI Tree Search Helpers ==========
    
    def enumerate_all_successors(self, state: GameState) -> List[Tuple[Direction, GameState, float]]:
        """
        Get ALL possible successor states with probabilities.
        
        This is the ONE-STOP method for Expectimax algorithm.
        
        Args:
            state: Current game state
            
        Returns:
            List of (direction, resulting_state, probability) tuples
            
        Process:
            For each valid move:
                Execute move -> Get all possible spawns -> Add to results
                
        Example:
            If 3 valid moves and average 8 empty cells after move:
            Classic mode: 3 × 8 × 2 = 48 successors
            Dynamic mode (late game): 3 × 8 × 12 = 288 successors
            
        Complexity:
            O(valid_moves × empty_cells × spawn_values)
        """
        successors = []
        
        # Get all valid move outcomes
        move_outcomes = self.move_engine.get_all_move_outcomes(state)
        
        for direction, after_move_state in move_outcomes.items():
            # Get all possible spawns
            possible_spawns = self.tile_spawner.get_all_possible_spawns(after_move_state)
            
            for spawn_state, spawn_prob in possible_spawns:
                successors.append((direction, spawn_state, spawn_prob))
        
        return successors
    
    def get_deterministic_move_states(self, state: GameState) -> Dict[Direction, GameState]:
        """
        Get states after moves but BEFORE random spawn.
        
        Useful for Monte Carlo Tree Search where you want to separate
        the deterministic move phase from the stochastic spawn phase.
        
        Args:
            state: Current game state
            
        Returns:
            Dictionary mapping valid directions to states after move (no spawn)
        """
        return self.move_engine.get_all_move_outcomes(state)
    
    def simulate_random_game(self, state: GameState, max_moves: int = 1000) -> GameState:
        """
        Play random moves until game over (for rollout evaluation).
        
        Args:
            state: Starting state
            max_moves: Maximum number of moves to simulate
            
        Returns:
            Final state after random play
            
        Usage:
            Used in Monte Carlo methods for quick position evaluation.
        """
        current_state = state
        moves_made = 0
        
        while not self.is_game_over(current_state) and moves_made < max_moves:
            valid_moves = self.move_engine.get_valid_moves(current_state)
            if not valid_moves:
                break
            
            # Choose random move
            direction = valid_moves[self.rng.integers(0, len(valid_moves))]
            current_state, _ = self.make_move(current_state, direction)
            moves_made += 1
        
        return current_state
