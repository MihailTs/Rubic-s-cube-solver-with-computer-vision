import kociemba
import random

class KociembaCubeSolver:
    """
    A Rubik's Cube solver using the Kociemba algorithm.
    
    Cube state notation:
    - 54 characters representing all stickers
    - Order: U(p), R(ight), F(ront), D(own), L(eft), B(ack)
    - Each face: 9 stickers in reading order (top-left to bottom-right)
    - Colors: U (White), R (Red), F (Green), D (Yellow), L (Orange), B (Blue)
    """
    
    def __init__(self, state=None):
        if state == None:
            self.state = 'U' * 9 + 'R' * 9 + 'F' * 9 + 'D' * 9 + 'L' * 9 + 'B' * 9
        else: 
            self.state = state

    def scramble(self, num_moves=25):
        """
        Generate a random scramble.
        Returns the scramble sequence as a string.
        """
        moves = ['U', 'D', 'L', 'R', 'F', 'B']
        modifiers = ['', "'", '2']
        
        scramble_sequence = []
        last_move = None
        
        for _ in range(num_moves):
            # Avoid consecutive moves on the same face
            available = [m for m in moves if m != last_move]
            move = random.choice(available)
            modifier = random.choice(modifiers)
            scramble_sequence.append(move + modifier)
            last_move = move
        
        return ' '.join(scramble_sequence)
    
    def solve(self, cube_state=None):
        """
        Solve the cube using Kociemba's algorithm.
        
        Args:
            cube_state: 54-character string representing cube state
                       If None, uses self.state
        
        Returns:
            Solution string with move sequence, or error message
        """
        if cube_state is None:
            cube_state = self.state
        
        solution = kociemba.solve(cube_state)
        return solution
    

    def validate_state(self, cube_state):
        """
        Validate that a cube state is valid.
        
        A valid cube state must:
        - Be 54 characters long
        - Contain exactly 9 of each color (U, R, F, D, L, B)
        """
        if len(cube_state) != 54:
            return False, "State must be 54 characters long"
        
        valid_colors = set('URFDLB')
        for color in cube_state:
            if color not in valid_colors:
                return False, f"Invalid color: {color}"
        
        # Check each color appears exactly 9 times
        for color in valid_colors:
            if cube_state.count(color) != 9:
                return False, f"Color {color} appears {cube_state.count(color)} times (should be 9)"
        
        return True, "Valid state"
    
    def print_state(self, cube_state=None):
        """Print a visual representation of the cube state."""
        if cube_state is None:
            cube_state = self.state
        
        # Extract faces
        U = cube_state[0:9]
        R = cube_state[9:18]
        F = cube_state[18:27]
        D = cube_state[27:36]
        L = cube_state[36:45]
        B = cube_state[45:54]
        
        # Print layout
        print("\nCube state:")
        print("      " + " ".join(U[0:3]))
        print("      " + " ".join(U[3:6]))
        print("      " + " ".join(U[6:9]))
        print()
        print(" ".join(L[0:3]) + "  " + " ".join(F[0:3]) + "  " + " ".join(R[0:3]) + "  " + " ".join(B[0:3]))
        print(" ".join(L[3:6]) + "  " + " ".join(F[3:6]) + "  " + " ".join(R[3:6]) + "  " + " ".join(B[3:6]))
        print(" ".join(L[6:9]) + "  " + " ".join(F[6:9]) + "  " + " ".join(R[6:9]) + "  " + " ".join(B[6:9]))
        print()
        print("      " + " ".join(D[0:3]))
        print("      " + " ".join(D[3:6]))
        print("      " + " ".join(D[6:9]))
        print()