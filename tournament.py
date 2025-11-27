"""
AI Tournament Runner

Runs a competition between different AI agents to benchmark performance.
"""

import argparse
import time
import statistics
import random
from typing import List, Dict, Any

from game_core import Game, SpawnMode
from run_ai import discover_players
from main import run_game


def run_tournament(games_per_agent: int = 10, mode: str = 'classic'):
    """
    Run tournament between all discovered agents.
    
    Args:
        games_per_agent: Number of games to run for each agent
        mode: Game mode ('classic' or 'dynamic')
    """
    print(f"🏆 Starting Tournament ({games_per_agent} games per agent, mode={mode})")
    print("=" * 60)
    
    # Discover agents
    all_players = discover_players()
    
    # Filter out HumanPlayer and base Player class
    agents = {
        name: info for name, info in all_players.items() 
        if 'Human' not in name and name != 'Player'
    }
    
    results = []
    
    for agent_name, (module_name, player_class, needs_game) in agents.items():
        print(f"\n🤖 Testing Agent: {agent_name}")
        
        scores = []
        max_tiles = []
        move_counts = []
        times = []
        wins_2048 = 0
        wins_65536 = 0
        
        start_time = time.time()
        
        for i in range(games_per_agent):
            # Random seed for each game
            seed = random.randint(0, 1000000)
            spawn_mode = SpawnMode.CLASSIC if mode == 'classic' else SpawnMode.DYNAMIC
            game = Game(spawn_mode=spawn_mode, seed=seed)
            
            # Create player
            if needs_game:
                # Default depth 3 for Expectimax variants
                if 'Expectimax' in agent_name or 'Stochastic' in agent_name:
                    player = player_class(game, depth=3)
                else:
                    player = player_class(game)
            else:
                player = player_class()
            
            # Run game
            game_start = time.time()
            final_state = run_game(game, player, verbose=False)
            game_duration = time.time() - game_start
            
            # Record stats
            scores.append(final_state.score)
            max_tiles.append(final_state.get_max_tile())
            move_counts.append(final_state.move_count)
            times.append(game_duration)
            
            if final_state.get_max_tile() >= 2048:
                wins_2048 += 1
            if final_state.get_max_tile() >= 65536:
                wins_65536 += 1
                
            print(f"  Game {i+1}/{games_per_agent}: Score={final_state.score}, Max={final_state.get_max_tile()}, Time={game_duration:.2f}s")
            
        total_time = time.time() - start_time
        
        # Calculate stats
        avg_score = statistics.mean(scores)
        max_score = max(scores)
        avg_max_tile = statistics.mean(max_tiles)
        best_tile = max(max_tiles)
        avg_moves = statistics.mean(move_counts)
        avg_time_per_game = statistics.mean(times)
        
        results.append({
            'Agent': agent_name,
            'Avg Score': avg_score,
            'Max Score': max_score,
            'Best Tile': best_tile,
            '2048+ %': (wins_2048 / games_per_agent) * 100,
            '65536+ %': (wins_65536 / games_per_agent) * 100,
            'Avg Time': avg_time_per_game
        })
    
    # Print Results Table
    print("\n\n🏆 TOURNAMENT RESULTS 🏆")
    print("=" * 95)
    
    # Sort by Avg Score descending
    results.sort(key=lambda x: x['Avg Score'], reverse=True)
    
    # Header
    headers = ['Agent', 'Avg Score', 'Max Score', 'Best Tile', '2048+ %', '65536+ %', 'Avg Time (s)']
    
    # Simple table formatting
    row_format = "{:<25} {:<12} {:<12} {:<10} {:<10} {:<10} {:<12}"
    print(row_format.format(*headers))
    print("-" * 95)
    
    for r in results:
        print(row_format.format(
            r['Agent'],
            f"{r['Avg Score']:.0f}",
            f"{r['Max Score']}",
            f"{r['Best Tile']}",
            f"{r['2048+ %']:.0f}%",
            f"{r['65536+ %']:.0f}%",
            f"{r['Avg Time']:.2f}"
        ))
    print("=" * 95)
    
    # Generate Plot
    plot_results(results)


def plot_results(results: List[Dict[str, Any]]):
    """Generate Score vs Time plot."""
    try:
        import matplotlib.pyplot as plt
        
        agents = [r['Agent'] for r in results]
        scores = [r['Avg Score'] for r in results]
        times = [r['Avg Time'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(times, scores, color='blue', s=100)
        
        # Add labels
        for i, agent in enumerate(agents):
            plt.annotate(agent, (times[i], scores[i]), xytext=(5, 5), textcoords='offset points')
            
        plt.title('AI Performance: Score vs Time')
        plt.xlabel('Average Time per Game (s)')
        plt.ylabel('Average Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plt.savefig('benchmark_results.png')
        print("\n📊 Plot saved to 'benchmark_results.png'")
        
    except ImportError:
        print("\n⚠️ Matplotlib not found. Skipping plot generation.")
        print("To enable plotting, install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AI Tournament')
    parser.add_argument('--games', type=int, default=5, help='Games per agent')
    parser.add_argument('--mode', choices=['classic', 'dynamic'], default='classic', help='Game mode')
    
    args = parser.parse_args()
    run_tournament(args.games, args.mode)
