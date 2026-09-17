"""
Dynamic Maze Pathfinding with A* Search Algorithm
Course: AI2002 Artificial Intelligence
Assignment 1: Search Algorithm Implementation

Group Members:
    - Shah Faisal (23I-0058)
    - Akbar Hussain (23I-3094)
    - Hamad Khan (23I-3095)

    Submitted to: Dr. Muhammad Bilal
    """

import heapq
import time
import os
import sys
from typing import List, Tuple, Dict, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import random


# ============================================================================
# DOMAIN-SPECIFIC ENUMS AND TYPES
# ============================================================================

class CellType(Enum):
    """Enumeration of possible cell types in the maze."""
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    PATH = 4
    EXPLORED = 5
    FRONTIER = 6


class Movement(Enum):
    """Possible movement directions."""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @classmethod
    def get_all_movements(cls):
        """Return all possible movements."""
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]


    # ============================================================================
    # NODE CLASS - Represents a state in the search space
    # ============================================================================

    @dataclass(order=True)
class Node:
    """
    Represents a node in the search tree.

    Attributes:
        f_score: Total estimated cost (g + h) - used for priority queue ordering
        g_score: Actual cost from start to this node
        h_score: Heuristic estimate to goal
        position: (x, y) coordinates in the grid
        parent: Parent node for path reconstruction
        action: Action taken to reach this node
        depth: Depth of node in search tree
        """
        f_score: float = field(init=False, compare=True)
        g_score: float = field(compare=False)
        h_score: float = field(compare=False)
        position: Tuple[int, int] = field(compare=False)
        parent: Optional['Node'] = field(default=None, compare=False)
        action: Optional[Movement] = field(default=None, compare=False)
        depth: int = field(default=0, compare=False)

    def __post_init__(self):
        """Calculate f_score after initialization."""
        self.f_score = self.g_score + self.h_score

    def get_path(self) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to this node.

        Returns:
            List of positions from start to this node
            """
            path = []
            current = self
            while current:
                path.append(current.position)
                current = current.parent
                return list(reversed(path))

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Node(pos={self.position}, g={self.g_score}, h={self.h_score}, f={self.f_score})"


    # ============================================================================
    # PROBLEM CLASS - Encapsulates the maze pathfinding problem
    # ============================================================================

class MazePathfindingProblem:
    """
    Represents the dynamic maze pathfinding problem.

    This class encapsulates:
        - State representation (grid with obstacles)
        - Actions (movement directions)
        - Transition model (how actions change state)
        - Goal test
        - Step costs
        - Heuristic functions
        """

        # Cell display characters for console output
        CELL_DISPLAY = {
            CellType.EMPTY: '·',
            CellType.OBSTACLE: '█',
            CellType.START: 'S',
            CellType.GOAL: 'G',
            CellType.PATH: '●',
            CellType.EXPLORED: '○',
            CellType.FRONTIER: '◉'
        }

        # ANSI color codes for console output
        COLORS = {
            CellType.EMPTY: '\033[90m',      # Gray
            CellType.OBSTACLE: '\033[91m',    # Red
            CellType.START: '\033[92m',       # Green
            CellType.GOAL: '\033[93m',         # Yellow
            CellType.PATH: '\033[94m',         # Blue
            CellType.EXPLORED: '\033[95m',     # Magenta
            CellType.FRONTIER: '\033[96m',     # Cyan
            'RESET': '\033[0m'
        }

    def __init__(self, width: int = 10, height: int = 10, obstacle_density: float = 0.2):
        """
        Initialize the maze pathfinding problem.

        Args:
            width: Grid width
            height: Grid height
            obstacle_density: Probability of obstacle in each cell (0.0 to 1.0)

            Raises:
                ValueError: If parameters are invalid
                """
                # Validate inputs
                if width < 2 or height < 2:
                    raise ValueError("Grid dimensions must be at least 2x2")
                if not 0 <= obstacle_density <= 1:
                    raise ValueError("Obstacle density must be between 0 and 1")

                self.width = width
                self.height = height
                self.obstacle_density = obstacle_density

                # Initialize grid with empty cells
                self.grid = [[CellType.EMPTY for _ in range(width)] for _ in range(height)]

                # Place obstacles randomly
                self._place_obstacles()

                # Set random start and goal positions
                self.start = self._get_random_empty_position()
                self.goal = self._get_random_empty_position(exclude=[self.start])

                # Mark start and goal on grid
                self._set_cell(self.start, CellType.START)
                self._set_cell(self.goal, CellType.GOAL)

                # Statistics tracking
                self.nodes_expanded = 0
                self.search_time = 0

    def _place_obstacles(self):
        """Randomly place obstacles in the grid based on obstacle density."""
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < self.obstacle_density:
                    self.grid[y][x] = CellType.OBSTACLE

    def _get_random_empty_position(self, exclude: List[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Get a random empty position in the grid.

        Args:
            exclude: List of positions to exclude

            Returns:
                Random empty position

                Raises:
                    RuntimeError: If no empty position found
                    """
                exclude = exclude or []
                    empty_positions = []

                    for y in range(self.height):
                        for x in range(self.width):
                            if self.grid[y][x] == CellType.EMPTY and (x, y) not in exclude:
                                empty_positions.append((x, y))

                                if not empty_positions:
                                    raise RuntimeError("No empty positions available in grid")

                                return random.choice(empty_positions)

    def _set_cell(self, position: Tuple[int, int], cell_type: CellType):
        """Set a cell in the grid."""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = cell_type

    def get_actions(self, position: Tuple[int, int]) -> List[Movement]:
        """
        Get valid actions from a given position.

        Args:
            position: Current position (x, y)

            Returns:
                List of valid movements
                """
                x, y = position
                valid_actions = []

                for movement in Movement.get_all_movements():
                    dx, dy = movement.value
                    new_x, new_y = x + dx, y + dy

                    # Check bounds
                    if 0 <= new_x < self.width and 0 <= new_y < self.height:
                        # Check if not obstacle
                        if self.grid[new_y][new_x] != CellType.OBSTACLE:
                            valid_actions.append(movement)

                            return valid_actions

    def transition(self, position: Tuple[int, int], action: Movement) -> Tuple[int, int]:
        """
        Apply an action to get a new position.

        Args:
            position: Current position
            action: Movement to apply

            Returns:
                New position after applying action
                """
    x, y = position
    dx, dy = action.value
    return (x + dx, y + dy)

    def step_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """
        Calculate the cost of moving from one position to another.

        Args:
            from_pos: Starting position
            to_pos: Target position

            Returns:
                Step cost (always 1.0 for grid movement)
                """
                return 1.0

    def heuristic(self, position: Tuple[int, int], heuristic_type: str = "manhattan") -> float:
        """
        Heuristic function estimating cost to goal.

        Args:
            position: Current position
            heuristic_type: Type of heuristic ("manhattan" or "euclidean")

            Returns:
                Estimated cost to goal

                Raises:
                    ValueError: If heuristic_type is invalid
                    """
        x1, y1 = position
        x2, y2 = self.goal

        if heuristic_type == "manhattan":
                        # Manhattan distance - admissible for 4-directional movement
                        return abs(x1 - x2) + abs(y1 - y2)
        elif heuristic_type == "euclidean":
                        # Euclidean distance - also admissible
                        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        else:
                        raise ValueError(f"Unknown heuristic type: {heuristic_type}")

    def is_goal(self, position: Tuple[int, int]) -> bool:
        """Check if a position is the goal."""
        return position == self.goal

    def reset_visualization(self):
        """Reset visualization markers on the grid."""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] in [CellType.EXPLORED, CellType.FRONTIER, CellType.PATH]:
                    if (x, y) == self.start:
                        self.grid[y][x] = CellType.START
                    elif (x, y) == self.goal:
                        self.grid[y][x] = CellType.GOAL
                    else:
                        self.grid[y][x] = CellType.EMPTY

    def mark_explored(self, position: Tuple[int, int]):
        """Mark a position as explored (for visualization)."""
        if position != self.start and position != self.goal:
            self._set_cell(position, CellType.EXPLORED)

    def mark_frontier(self, position: Tuple[int, int]):
        """Mark a position as frontier (for visualization)."""
        if position != self.start and position != self.goal:
            self._set_cell(position, CellType.FRONTIER)

    def mark_path(self, path: List[Tuple[int, int]]):
        """Mark the solution path on the grid."""
        for position in path:
            if position != self.start and position != self.goal:
                self._set_cell(position, CellType.PATH)

    def display(self, show_colors: bool = True):
        """
        Display the current grid state.

        Args:
            show_colors: Whether to use ANSI colors
            """
            # Print column numbers
    print("\n    ", end="")
    for x in range(self.width):
                print(f"{x:2} ", end="")
                print()

                # Print top border
                print("   ┌" + "───" * self.width + "┐")

                # Print grid rows
                for y in range(self.height):
                    print(f"{y:2} │", end="")
                    for x in range(self.width):
                        cell = self.grid[y][x]

                        if show_colors:
                            print(f"{self.COLORS[cell]}", end="")
                            print(f" {self.CELL_DISPLAY[cell]} ", end="")
                            print(f"{self.COLORS['RESET']}", end="")
                        else:
                            print(f" {self.CELL_DISPLAY[cell]} ", end="")
                            print("│")

                            # Print bottom border
                            print("   └" + "───" * self.width + "┘")
                            print()


                            # ============================================================================
                            # A* SEARCH ALGORITHM IMPLEMENTATION
                            # ============================================================================

class AStarSearch:
    """
    A* search algorithm implementation for maze pathfinding.

    Features:
        - Optimal pathfinding with admissible heuristics
        - Priority queue implementation using heapq
        - Explored set for graph search
        - Detailed statistics tracking
        - Visualization support
        """

    def __init__(self, problem: MazePathfindingProblem, heuristic_type: str = "manhattan"):
        """
        Initialize A* search.

        Args:
            problem: The maze pathfinding problem
            heuristic_type: Type of heuristic to use
            """
        self.problem = problem
        self.heuristic_type = heuristic_type
        # Search statistics
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier_size = 0
        self.search_time = 0

    def search(self, visualize: bool = True) -> Optional[Node]:
        """
        Perform A* search to find optimal path.

        Args:
            visualize: Whether to visualize search progress

            Returns:
                Goal node if path found, None otherwise
                """
                # Reset visualization if needed
        if visualize:
                    self.problem.reset_visualization()

                    # Start timing
                    start_time = time.time()

                    # Initialize start node
                    start_h = self.problem.heuristic(self.problem.start, self.heuristic_type)
                    start_node = Node(
                        g_score=0,
                        h_score=start_h,
                        position=self.problem.start,
                        parent=None,
                        action=None,
                        depth=0
                    )

                    # Priority queue (min-heap) for frontier, ordered by f_score
                    # Heap elements: (f_score, tie_breaker, node)
                    # tie_breaker ensures deterministic behavior for equal f_scores
                    frontier = []
                    tie_breaker = 0
                    heapq.heappush(frontier, (start_node.f_score, tie_breaker, start_node))
                    self.nodes_generated += 1

                    # Dictionary for quick lookup of nodes in frontier
                    frontier_dict = {self.problem.start: start_node}

                    # Explored set (closed list)
                    explored = set()

                    # For visualization
                    if visualize:
                        self.problem.display()
                        print("Searching... Press Ctrl+C to skip visualization\n")
                        time.sleep(0.5)

                        try:
                            while frontier:
                                # Update max frontier size statistic
                                self.max_frontier_size = max(self.max_frontier_size, len(frontier))

                                # Pop node with lowest f_score
                                _, _, current_node = heapq.heappop(frontier)
                                current_pos = current_node.position

                                # Remove from frontier dictionary
                                if current_pos in frontier_dict:
                                    del frontier_dict[current_pos]

                                    # Visualize current expansion
                                    if visualize:
                                        self.problem.mark_explored(current_pos)
                                        self.problem.display()
                                        print(f"Expanding: {current_pos} (f={current_node.f_score:.1f})")
                                        time.sleep(0.1)

                                        # Goal test
                                        if self.problem.is_goal(current_pos):
                                            self.search_time = time.time() - start_time
                                            self.nodes_expanded = len(explored)

                                            # Visualize final path
                                            if visualize:
                                                path = current_node.get_path()
                                                self.problem.mark_path(path)
                                                self.problem.display()
                                                print("\n" + "="*50)
                                                print("GOAL REACHED!")
                                                print("="*50)

                                                return current_node

                                            # Add to explored set
                                            explored.add(current_pos)
                                            self.nodes_expanded += 1

                                            # Generate successors
                                            for action in self.problem.get_actions(current_pos):
                                                new_pos = self.problem.transition(current_pos, action)

                                                # Skip if already explored
                                                if new_pos in explored:
                                                    continue

                                                # Calculate costs
                                                tentative_g = current_node.g_score + self.problem.step_cost(current_pos, new_pos)
                                                h = self.problem.heuristic(new_pos, self.heuristic_type)

                                                # Check if new_pos is in frontier
                                                if new_pos in frontier_dict:
                                                    existing_node = frontier_dict[new_pos]

                                                    # If we found a better path
                                                    if tentative_g < existing_node.g_score:
                                                        # Update existing node
                                                        existing_node.g_score = tentative_g
                                                        existing_node.h_score = h
                                                        existing_node.parent = current_node
                                                        existing_node.action = action
                                                        existing_node.depth = current_node.depth + 1

                                                        # Need to re-heapify - we'll just push new and ignore old
                                                        # (lazy deletion approach)
                                                        tie_breaker += 1
                                                        heapq.heappush(frontier, (existing_node.f_score, tie_breaker, existing_node))
                                                        self.nodes_generated += 1
                                                    else:
                                                        # Create new node
                                                        new_node = Node(
                                                            g_score=tentative_g,
                                                            h_score=h,
                                                            position=new_pos,
                                                            parent=current_node,
                                                            action=action,
                                                            depth=current_node.depth + 1
                                                        )

                                                        # Add to frontier
                                                        tie_breaker += 1
                                                        heapq.heappush(frontier, (new_node.f_score, tie_breaker, new_node))
                                                        frontier_dict[new_pos] = new_node
                                                        self.nodes_generated += 1

                                                        # Visualize frontier addition
                                                        if visualize:
                                                            self.problem.mark_frontier(new_pos)
                                                            self.problem.display()
                                                            print(f"Added to frontier: {new_pos} (f={new_node.f_score:.1f})")
                                                            time.sleep(0.05)

                        except KeyboardInterrupt:
                            print("\nVisualization interrupted. Completing search in background...")

                            # No path found
                            self.search_time = time.time() - start_time
                            self.nodes_expanded = len(explored)
                            return None

    def print_statistics(self):
        """Print detailed search statistics."""
        print("\n" + "="*50)
        print("SEARCH STATISTICS")
        print("="*50)
        print(f"Nodes Expanded: {self.nodes_expanded}")
        print(f"Nodes Generated: {self.nodes_generated}")
        print(f"Max Frontier Size: {self.max_frontier_size}")
        print(f"Search Time: {self.search_time:.3f} seconds")
        print(f"Heuristic Used: {self.heuristic_type.capitalize()}")
        print("="*50)


        # ============================================================================
        # INTERACTIVE CONSOLE APPLICATION
        # ============================================================================

class PathfindingApp:
    """
    Interactive console application for maze pathfinding.

    Features:
        - Menu-driven interface
        - Configurable maze generation
        - Multiple heuristic options
        - Visualization toggle
        - Path display and statistics
        """

    def __init__(self):
        self.problem = None
        self.search_algorithm = None
        self.solution = None

    def clear_screen(self):
        """Clear the console screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """Print application header."""
        print("="*60)
        print("          DYNAMIC MAZE PATHFINDING WITH A* SEARCH")
        print("="*60)
        print("Group: Shah Faisal (23I-0058), Akbar Hussain (23I-3094), Hamad Khan (23I-3095)")
        print("="*60)
        print()

    def get_int_input(self, prompt: str, min_val: int = None, max_val: int = None) -> int:
        """
        Get integer input from user with validation.

        Args:
            prompt: Input prompt
            min_val: Minimum allowed value
            max_val: Maximum allowed value

            Returns:
                Validated integer input
                """
                while True:
                    try:
                        value = int(input(prompt))
                        if min_val is not None and value < min_val:
                            print(f"Value must be at least {min_val}")
                            continue
                        if max_val is not None and value > max_val:
                            print(f"Value must be at most {max_val}")
                            continue
                        return value
                    except ValueError:
                        print("Please enter a valid integer")

    def get_float_input(self, prompt: str, min_val: float = None, max_val: float = None) -> float:
        """Get float input from user with validation."""
        while True:
            try:
                value = float(input(prompt))
                if min_val is not None and value < min_val:
                    print(f"Value must be at least {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Value must be at most {max_val}")
                    continue
                return value
            except ValueError:
                print("Please enter a valid number")

    def get_yes_no(self, prompt: str) -> bool:
        """Get yes/no input from user."""
        while True:
            response = input(prompt).strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            print("Please enter 'y' or 'n'")

    def configure_maze(self):
        """Configure maze parameters."""
        self.clear_screen()
        self.print_header()

        print("MAZE CONFIGURATION")
        print("-" * 40)

        width = self.get_int_input("Enter grid width (5-50): ", 5, 50)
        height = self.get_int_input("Enter grid height (5-50): ", 5, 50)
        density = self.get_float_input("Enter obstacle density (0.0-0.4): ", 0.0, 0.4)

        try:
            self.problem = MazePathfindingProblem(width, height, density)
            print("\nMaze generated successfully!")
            self.problem.display()
        except Exception as e:
            print(f"\nError generating maze: {e}")
            input("\nPress Enter to continue...")
            self.configure_maze()

    def run_search(self):
        """Run A* search on the configured maze."""
        if not self.problem:
            print("\nPlease configure a maze first!")
            input("Press Enter to continue...")
            return

        self.clear_screen()
        self.print_header()

        print("SEARCH CONFIGURATION")
        print("-" * 40)

        # Select heuristic
        print("\nHeuristic options:")
        print("1. Manhattan distance (admissible)")
        print("2. Euclidean distance (admissible)")
        heuristic_choice = self.get_int_input("Choose heuristic (1-2): ", 1, 2)
        heuristic = "manhattan" if heuristic_choice == 1 else "euclidean"

        # Visualization option
        visualize = self.get_yes_no("Enable step-by-step visualization? (y/n): ")

        # Create and run search
        self.search_algorithm = AStarSearch(self.problem, heuristic)

        print("\n" + "="*50)
        print("STARTING A* SEARCH")
        print("="*50)
        print(f"Start: {self.problem.start}")
        print(f"Goal: {self.problem.goal}")
        print("="*50)

        self.solution = self.search_algorithm.search(visualize)

        if self.solution:
            path = self.solution.get_path()
            print(f"\nPath found! Length: {len(path)-1} steps")
            print(f"Path: {' -> '.join([str(p) for p in path])}")
        else:
            print("\nNo path found to goal!")

            self.search_algorithm.print_statistics()

            input("\nPress Enter to continue...")

    def show_menu(self):
        """Display main menu and handle user choice."""
        while True:
            self.clear_screen()
            self.print_header()

            print("MAIN MENU")
            print("-" * 40)
            print("1. Configure New Maze")
            print("2. Run A* Search")
            print("3. Display Current Maze")
            print("4. Random Maze Example")
            print("5. Exit")
            print("-" * 40)

            if self.problem:
                print(f"Current Maze: {self.problem.width}x{self.problem.height}, "
                f"Density: {self.problem.obstacle_density:.1f}")
            else:
                print("No maze configured")
                print("-" * 40)

                choice = self.get_int_input("Enter your choice (1-5): ", 1, 5)

                if choice == 1:
                    self.configure_maze()
                elif choice == 2:
                    self.run_search()
                elif choice == 3:
                    if self.problem:
                        self.clear_screen()
                        self.print_header()
                        self.problem.display()
                        input("\nPress Enter to continue...")
                    else:
                        print("\nNo maze configured!")
                        input("Press Enter to continue...")
                elif choice == 4:
                    self.run_example()
                elif choice == 5:
                    print("\nThank you for using the Pathfinding Application!")
                    print("Goodbye!")
                    sys.exit(0)

    def run_example(self):
        """Run a predefined example to demonstrate functionality."""
        self.clear_screen()
        self.print_header()

        print("RUNNING EXAMPLE DEMONSTRATION")
        print("="*50)

        # Create a 10x10 maze with 20% obstacles
        self.problem = MazePathfindingProblem(10, 10, 0.2)
        print("Generated 10x10 maze with 20% obstacles:")
        self.problem.display()

        # Run A* with Manhattan heuristic
        print("\nRunning A* with Manhattan heuristic...")
        self.search_algorithm = AStarSearch(self.problem, "manhattan")

        print(f"Start: {self.problem.start}")
        print(f"Goal: {self.problem.goal}")
        print("="*50)

        self.solution = self.search_algorithm.search(visualize=False)

        if self.solution:
            path = self.solution.get_path()
            print(f"\n✓ Path found! Length: {len(path)-1} steps")
            print(f"Path: {path}")

            # Mark and display path
            self.problem.reset_visualization()
            self.problem.mark_path(path)
            self.problem.display()
        else:
            print("\n✗ No path found to goal!")
            self.problem.display()

            self.search_algorithm.print_statistics()

            input("\nPress Enter to continue...")


            # ============================================================================
            # MAIN ENTRY POINT
            # ============================================================================

def main():
    """Main entry point for the application."""
    app = PathfindingApp()

    try:
        app.show_menu()
    except KeyboardInterrupt:
        print("\n\nApplication terminated by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
import traceback
traceback.print_exc()

print("\nGoodbye!")


if __name__ == "__main__":
    main()