"""
Human Player with Tkinter GUI

Simple windowed interface for playing 65536 game.
Controls: Arrow keys (↑ ↓ ← →)
"""

import tkinter as tk
from tkinter import messagebox
from player_interface import Player
from game_core import GameState, Direction, MoveResult, Game, SpawnMode


# Color scheme for tiles
TILE_COLORS = {
    0: "#cdc1b4",
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
    4096: "#3c3a32",
    8192: "#3c3a32",
    16384: "#3c3a32",
    32768: "#3c3a32",
    65536: "#3c3a32",
}

TILE_TEXT_COLORS = {
    0: "#776e65",
    2: "#776e65",
    4: "#776e65",
    8: "#f9f6f2",
    16: "#f9f6f2",
}


class HumanPlayer(Player):
    """Human player with Tkinter GUI."""
    
    def __init__(self, game: Game):
        self.game = game
        self.window = None
        self.tiles = []
        self.score_label = None
        self.move_label = None
        self.next_move = None
        self.waiting_for_move = False
    
    def get_move(self, state: GameState) -> Direction:
        """Wait for human input via keyboard."""
        if not self.window:
            self._create_window()
        
        self._update_display(state)
        
        # Wait for keyboard input
        self.waiting_for_move = True
        self.next_move = None
        
        while self.next_move is None and self.waiting_for_move:
            self.window.update()
        
        return self.next_move
    
    def on_move_result(self, result: MoveResult) -> None:
        """Update display after move."""
        if self.window:
            self._update_display(result.state)
            self.window.update()
    
    def on_game_over(self, state: GameState) -> None:
        """Show game over message."""
        if not self.window or not self.window.winfo_exists():
            return
            
        self._update_display(state)
        
        if state.won:
            msg = f"🎉 YOU WON!\n\nReached 65536!\nFinal Score: {state.score:,}\nMoves: {state.move_count}"
        else:
            msg = f"Game Over!\n\nFinal Score: {state.score:,}\nMax Tile: {state.get_max_tile()}\nMoves: {state.move_count}"
        
        try:
            messagebox.showinfo("Game Over", msg)
            self.window.quit()
            self.window.destroy()
        except:
            pass  # Window already closed
    
    def reset(self) -> None:
        """Reset for new game."""
        self.next_move = None
        self.waiting_for_move = False
    
    def _create_window(self):
        """Create the game window."""
        self.window = tk.Tk()
        self.window.title("65536 Game")
        self.window.configure(bg="#faf8ef")
        
        # Header
        header = tk.Frame(self.window, bg="#faf8ef")
        header.pack(pady=20)
        
        title = tk.Label(header, text="65536", font=("Arial", 36, "bold"),
                        bg="#faf8ef", fg="#776e65")
        title.pack(side=tk.LEFT, padx=20)
        
        info_frame = tk.Frame(header, bg="#faf8ef")
        info_frame.pack(side=tk.LEFT, padx=20)
        
        self.score_label = tk.Label(info_frame, text="Score: 0",
                                    font=("Arial", 16), bg="#bbada0",
                                    fg="#ffffff", padx=15, pady=5)
        self.score_label.pack()
        
        self.move_label = tk.Label(info_frame, text="Moves: 0",
                                   font=("Arial", 12), bg="#faf8ef",
                                   fg="#776e65", pady=2)
        self.move_label.pack()
        
        # Instructions
        instruction = tk.Label(self.window, text="Use Arrow Keys to Play",
                             font=("Arial", 12), bg="#faf8ef", fg="#776e65")
        instruction.pack(pady=5)
        
        # Game grid
        grid_frame = tk.Frame(self.window, bg="#bbada0", padx=10, pady=10)
        grid_frame.pack(pady=10)
        
        self.tiles = []
        for i in range(4):
            row = []
            for j in range(4):
                tile = tk.Label(grid_frame, text="", font=("Arial", 32, "bold"),
                              width=5, height=2, bg=TILE_COLORS[0],
                              fg=TILE_TEXT_COLORS.get(0, "#f9f6f2"),
                              relief=tk.RAISED, borderwidth=2)
                tile.grid(row=i, column=j, padx=5, pady=5)
                row.append(tile)
            self.tiles.append(row)
        
        # Bind keyboard
        self.window.bind("<Up>", lambda e: self._on_key(Direction.UP))
        self.window.bind("<Down>", lambda e: self._on_key(Direction.DOWN))
        self.window.bind("<Left>", lambda e: self._on_key(Direction.LEFT))
        self.window.bind("<Right>", lambda e: self._on_key(Direction.RIGHT))
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _update_display(self, state: GameState):
        """Update the visual display."""
        if not self.window:
            return
        
        # Update tiles
        for i in range(4):
            for j in range(4):
                value = int(state.grid[i, j])
                tile = self.tiles[i][j]
                
                if value == 0:
                    tile.config(text="", bg=TILE_COLORS[0],
                              fg=TILE_TEXT_COLORS.get(0, "#f9f6f2"))
                else:
                    tile.config(text=str(value),
                              bg=TILE_COLORS.get(value, "#3c3a32"),
                              fg=TILE_TEXT_COLORS.get(value, "#f9f6f2"))
        
        # Update score and moves
        self.score_label.config(text=f"Score: {state.score:,}")
        self.move_label.config(text=f"Moves: {state.move_count}")
    
    def _on_key(self, direction: Direction):
        """Handle keyboard input."""
        if self.waiting_for_move:
            self.next_move = direction
            self.waiting_for_move = False
    
    def _on_close(self):
        """Handle window close."""
        self.waiting_for_move = False
        self.next_move = None
        if self.window:
            try:
                self.window.quit()
                self.window.destroy()
            except:
                pass
            self.window = None


if __name__ == "__main__":
    """Play a game as human."""
    from main import run_game
    
    print("Starting 65536 game...")
    print("Use arrow keys to play!")
    print()
    
    game = Game(spawn_mode=SpawnMode.CLASSIC)
    player = HumanPlayer(game)
    
    try:
        final = run_game(game, player)
        print(f"\nFinal: {final.score:,} points, max tile {final.get_max_tile()}")
    except Exception as e:
        print(f"Game ended: {e}")
