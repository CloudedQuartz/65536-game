"""
Board Evaluation Functions for AI Heuristics

This module provides evaluation functions for assessing 2048/65536 game states.
These are pure functions that take a GameState and return numeric scores.

All functions are independent of the core game logic - they're tools for AI
to evaluate positions, not part of the game mechanics.
"""

import numpy as np
from game_core import GameState


def get_smoothness(state: GameState) -> float:
    """
    Calculate board smoothness.
    
    Definition:
        Sum of absolute differences between adjacent tiles.
        Lower is better (tiles are more similar to neighbors).
        
    Formula:
        smoothness = Σ |tile[i,j] - tile[neighbor]| for all adjacent pairs
        
    Returns:
        Smoothness score (lower = smoother board)
        
    Complexity: O(1) - fixed 4x4 grid
    
    Interpretation:
        - Lower values = tiles are similar to neighbors (good)
        - Higher values = large differences between neighbors (bad)
        - Smooth boards are easier to merge
        
    Example:
        [[2, 2, 0, 0],
         [2, 2, 0, 0],  -> Low smoothness (similar neighbors)
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    """
    smoothness = 0.0
    grid = state.grid
    
    for i in range(4):
        for j in range(4):
            if grid[i, j] != 0:
                value = grid[i, j]
                
                # Check right neighbor
                if j < 3 and grid[i, j+1] != 0:
                    smoothness += abs(value - grid[i, j+1])
                
                # Check down neighbor
                if i < 3 and grid[i+1, j] != 0:
                    smoothness += abs(value - grid[i+1, j])
    
    return smoothness


def get_monotonicity(state: GameState) -> float:
    """
    Calculate board monotonicity.
    
    Definition:
        Measures how monotonic (consistently increasing or decreasing)
        each row and column is. Higher is better.
        
    Algorithm:
        For each row/column, check if values are monotonically
        increasing or decreasing. Sum up monotonic score.
        
    Returns:
        Monotonicity score (higher = more monotonic)
        
    Complexity: O(1) - fixed 4x4 grid
    
    Interpretation:
        - Higher values = board has monotonic rows/columns (good)
        - Lower values = board is chaotic (bad)
        - Monotonic boards follow a clear strategy (e.g., building up in corner)
        
    Example:
        [[128, 64, 32, 16],
         [ 64, 32, 16,  8],  -> High monotonicity (decreasing left-to-right)
         [ 32, 16,  8,  4],
         [ 16,  8,  4,  2]]
    """
    def monotonic_score(line: np.ndarray) -> float:
        """Score single line for monotonicity."""
        non_zero = line[line != 0]
        if len(non_zero) <= 1:
            return 0.0
        
        # Check if increasing or decreasing
        increasing = np.all(non_zero[:-1] <= non_zero[1:])
        decreasing = np.all(non_zero[:-1] >= non_zero[1:])
        
        if increasing or decreasing:
            return float(len(non_zero))
        return 0.0
    
    score = 0.0
    grid = state.grid
    
    # Check rows
    for i in range(4):
        score += monotonic_score(grid[i, :])
    
    # Check columns
    for j in range(4):
        score += monotonic_score(grid[:, j])
    
    return score


def get_merge_potential(state: GameState) -> int:
    """
    Count number of possible merges.
    
    Definition:
        Number of adjacent tile pairs with the same value that could merge.
        
    Returns:
        Count of possible merges
        
    Complexity: O(1) - fixed 4x4 grid
    
    Interpretation:
        - Higher values = more merging opportunities (good)
        - 0 = no possible merges (may indicate game over soon)
        
    Example:
        [[2, 2, 4, 4],
         [2, 8, 8, 4],  -> merge_potential = 4
         [0, 0, 0, 0],     (2-2, 4-4, 8-8, 4-4 if moved)
         [0, 0, 0, 0]]
    """
    merges = 0
    grid = state.grid
    
    for i in range(4):
        for j in range(4):
            if grid[i, j] != 0:
                value = grid[i, j]
                
                # Check right neighbor
                if j < 3 and grid[i, j+1] == value:
                    merges += 1
                
                # Check down neighbor
                if i < 3 and grid[i+1, j] == value:
                    merges += 1
    
    return merges


def get_corner_weight(state: GameState) -> float:
    """
    Weight of tiles in corners.
    
    Definition:
        Weighted sum where corner tiles have higher weights.
        Encourages keeping high-value tiles in corners.
        
    Formula:
        corner_weight = Σ tile_value × corner_multiplier
        Corner multipliers: 4.0 for corners, 1.0 for others
        
    Returns:
        Corner-weighted sum
        
    Complexity: O(1) - fixed 4x4 grid
    
    Interpretation:
        - Higher values = high tiles are in corners (good strategy)
        - Keeping highest tile in corner is a common 2048/65536 strategy
        
    Example:
        [[512, 256,  8,  4],
         [ 64,  32,  4,  2],  -> High corner weight (512 in top-left corner)
         [  8,   4,  2,  0],
         [  4,   2,  0,  0]]
    """
    weight = 0.0
    grid = state.grid
    
    # Corners have 4x weight
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    for i, j in corners:
        weight += grid[i, j] * 4.0
    
    # Rest have 1x weight
    for i in range(4):
        for j in range(4):
            if (i, j) not in corners and grid[i, j] != 0:
                weight += grid[i, j] * 1.0
    
    return weight


def count_distinct_tiles(state: GameState) -> int:
    """
    Count number of distinct tile values.
    
    Returns:
        Number of different non-zero values on board
        
    Complexity: O(1) - fixed 4x4 grid
    
    Interpretation:
        - Lower values = tiles are consolidated (good)
        - Higher values = many different tiles (bad, harder to merge)
        
    Example:
        [[2, 2, 4, 8],
         [2, 4, 8, 0],  -> distinct_tiles = 3 (values: 2, 4, 8)
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    """
    unique_values = np.unique(state.grid)
    # Exclude 0
    return len(unique_values[unique_values != 0])


def is_symmetric(state: GameState) -> bool:
    """
    Check if board has horizontal or vertical symmetry.
    
    Returns:
        True if board is symmetric, False otherwise
        
    Complexity: O(1) - fixed 4x4 grid
    
    Usage:
        Some AI algorithms can exploit symmetry to reduce search space.
    """
    grid = state.grid
    
    # Check horizontal symmetry
    h_sym = np.array_equal(grid, np.flip(grid, axis=0))
    
    # Check vertical symmetry
    v_sym = np.array_equal(grid, np.flip(grid, axis=1))
    
    return h_sym or v_sym


def get_gradient_score(state: GameState) -> float:
    """
    Calculate score based on a gradient weight matrix.
    
    This forces a 'snake' pattern where high values are pushed to one corner
    and decrease monotonically along a winding path.
    
    Weights (targeting top-left):
        [[ 65536, 32768, 16384, 8192 ],
         [   512,  1024,  2048, 4096 ],
         [   256,   128,    64,   32 ],
         [     2,     4,     8,   16 ]]
         
    Returns:
        Weighted sum of tile values
    """
    # Gradient weights favoring top-left corner in snake pattern
    weights = np.array([
        [65536, 32768, 16384, 8192],
        [  512,  1024,  2048, 4096],
        [  256,   128,    64,   32],
        [    2,     4,     8,   16]
    ], dtype=np.float64)
    
    return float(np.sum(state.grid * weights))


# Convenience function for combined evaluation
def evaluate_state(state: GameState, weights: dict = None) -> float:
    """
    Evaluate a state using weighted combination of heuristics.
    
    Args:
        state: GameState to evaluate
        weights: Dictionary of weights for each heuristic
                Default weights provided if None
    
    Returns:
        Combined evaluation score
    """
    if weights is None:
        # Default weights
        weights = {
            'gradient': 1.0,      # Strong snake pattern
            'empty_cells': 100.0, # Survival
            'smoothness': -0.1,   # Local mergeability
        }
    
    score = 0.0
    
    # Only calculate metrics that have non-zero weights
    # This optimization is crucial for search speed
    
    if weights.get('gradient', 0) != 0:
        score += get_gradient_score(state) * weights['gradient']
        
    if weights.get('empty_cells', 0) != 0:
        score += state.get_empty_count() * weights['empty_cells']
        
    if weights.get('position', 0) != 0:
        # 'position' is deprecated, mapped to gradient
        score += get_gradient_score(state) * (weights['position'] * 0.5)
    
    if weights.get('monotonicity', 0) != 0:
        score += get_monotonicity(state) * weights['monotonicity']
    
    if weights.get('smoothness', 0) != 0:
        score += get_smoothness(state) * weights['smoothness']
    
    if weights.get('merge_potential', 0) != 0:
        score += get_merge_potential(state) * weights['merge_potential']
    
    if weights.get('corner', 0) != 0:
        score += get_corner_weight(state) * weights['corner']
    
    return score
