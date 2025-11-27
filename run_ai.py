"""
Universal AI Runner

Automatically discovers and runs any AI implementation that inherits from Player.
Usage:
    python run_ai.py                    # List available players
    python run_ai.py RandomAIPlayer
    python run_ai.py HeuristicAI --games 10
    python run_ai.py ExpectimaxAI --depth 3 --games 5
    python run_ai.py HumanPlayer
"""

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import Dict, Type

from game_core import Game, SpawnMode
from player_interface import Player
from main import run_game, run_benchmark, print_benchmark_results


def discover_players() -> Dict[str, tuple]:
    """
    Auto-discover all Player subclasses by scanning Python files.
    
    Returns:
        Dict mapping player name to (module_path, class_object, needs_game_arg)
    """
    players = {}
    root = Path(__file__).parent
    
    # Find all .py files in root and subdirectories
    python_files = []
    
    # Root level .py files
    for file in root.glob('*.py'):
        if file.stem not in ['__init__', 'setup', 'run_ai']:
            python_files.append(file.stem)
    
    # Examples directory
    examples_dir = root / 'examples'
    if examples_dir.exists():
        for file in examples_dir.glob('*.py'):
            if file.stem != '__init__':
                python_files.append(f'examples.{file.stem}')
    
    # Agents directory
    agents_dir = root / 'agents'
    if agents_dir.exists():
        for file in agents_dir.glob('*.py'):
            if file.stem != '__init__':
                python_files.append(f'agents.{file.stem}')
    
    # Try importing each module and finding Player subclasses
    for module_name in python_files:
        try:
            module = importlib.import_module(module_name)
            
            # Find all classes that inherit from Player
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Player) and obj is not Player:
                    # Check if __init__ needs game argument
                    sig = inspect.signature(obj.__init__)
                    params = list(sig.parameters.keys())
                    needs_game = 'game' in params
                    
                    players[name] = (module_name, obj, needs_game)
        except Exception as e:
            pass  # Skip modules that fail to import
    
    return players



def main():
    # Discover available players
    available_players = discover_players()
    
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        print("Available Players:")
        for name in sorted(available_players.keys()):
            print(f"  - {name}")
        print("\nUsage:")
        print("  python run_ai.py <PlayerName> [options]")
        print("\nOptions:")
        print("  --mode {classic,dynamic}  Spawn mode (default: classic)")
        print("  --seed INT                Random seed (default: 42)")
        print("  --games INT               Number of games (default: 1)")
        print("  --depth INT               Search depth for Expectimax (default: 2)")
        print("  --verbose                 Print game state after each move")
        return
    
    parser = argparse.ArgumentParser(description='Run 65536 game with any AI player')
    parser.add_argument('player', help='Player class name')
    parser.add_argument('--mode', choices=['classic', 'dynamic'], default='classic',
                       help='Spawn mode (default: classic)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--games', type=int, default=1,
                       help='Number of games to run (default: 1)')
    parser.add_argument('--depth', type=int, default=2,
                       help='Search depth for Expectimax (default: 2)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print game state after each move')
    
    args = parser.parse_args()
    
    # Find player
    if args.player not in available_players:
        print(f"Error: Player '{args.player}' not found.")
        print(f"\nAvailable players: {', '.join(sorted(available_players.keys()))}")
        return
    
    module_name, player_class, needs_game = available_players[args.player]
    
    # Create game
    mode = SpawnMode.CLASSIC if args.mode == 'classic' else SpawnMode.DYNAMIC
    game = Game(spawn_mode=mode, seed=args.seed)
    
    # Create player factory
    def create_player():
        if needs_game:
            # Check if it's ExpectimaxAI (needs depth parameter)
            if 'Expectimax' in args.player:
                return player_class(game, depth=args.depth)
            else:
                return player_class(game)
        else:
            # Players that don't need game (shouldn't exist, but handle it)
            return player_class()
    
    player_name = args.player
    if 'Expectimax' in args.player:
        player_name += f" (depth={args.depth})"
    
    # Special handling for HumanPlayer (GUI mode)
    if 'Human' in args.player:
        print(f"Starting game with {player_name}...")
        print("Use arrow keys to play!")
        player = create_player()
        final = run_game(game, player)
        print(f"\nFinal: {final.score:,} points, max tile {final.get_max_tile()}")
        return
    
    # Run game(s) for AI
    if args.games == 1:
        print(f"Running 1 game with {player_name}...")
        player = create_player()
        final = run_game(game, player, verbose=args.verbose)
        print(f"\nFinal: {final.score:,} points, max tile {final.get_max_tile()}")
    else:
        print(f"Running {args.games} games with {player_name}...")
        stats = run_benchmark(
            game=game,
            player_factory=create_player,
            num_games=args.games,
            verbose_every=max(1, args.games // 5)
        )
        print()
        print_benchmark_results(stats)


if __name__ == "__main__":
    main()
