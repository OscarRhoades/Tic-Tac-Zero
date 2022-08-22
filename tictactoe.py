import numpy as np
import math as m
from numba import njit, jit


@njit
def win(board):
    for row in board:
        if np.all(row == row[0]) and row[0] != 0:
            # print("ROW")
            return True

    for col in board.T:
        if np.all(col == col[0]) and col[0] != 0:
            # print("COL")
            return True
        
    if (int(board[0][0]) == int(board[1][1]) and int(board[2][2]) == board[0][0]) and (board[0][0] != 0):
        # print("LEFT")
        return True
    
    if (int(board[0][2]) == int(board[1][1]) and int(board[2][0]) == board[1][1]) and (board[1][1] != 0):
        
        
        return True
        
    return False


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3,3))
        # self.turn = True
    
    def local_win(self):
        return win(self.board)
              
    def local_move(self,x,y, move_value):
        self.board[int(x)][int(y)] = move_value
        
    
    def mark_complete(self, region_value):
        
        for x in range(3):
            for y in range(3):
                self.board[x][y] = region_value
                
    def get_c(self,x,y):
        return self.board[x][y]
    
    def show(self):
        print(self.board)
    
    
        
        
class Ultimate:
    def __init__(self):
        self.x_turn = True
        # true if it is the x players turn
        self.taken_squares = []
        
        
        self.board = []
        for build_y in range(3):
            row = []
            for build_x in range(3):
                row.append(TicTacToe())
            self.board.append(row)
                
                
                
                
        self.game_board = np.zeros((3,3))
        
    
    def win(self):
        if win(self.game_board) == True:
            print(self.game_board)
        return win(self.game_board)
    
    
    
    def draw(self):
        if win(self.game_board) == False and self.legal_moves() == []:
            print(self.game_board)
            return True
            
    
    
    def global_move(self,x, y):
        x_global = m.floor(x / 3)
        y_global = m.floor(y / 3)
        
        x_local = x % 3
        y_local = y % 3
        
        
        
        move_value = 1 if self.x_turn else -1
        self.board[x_global][y_global].local_move(x_local, y_local, move_value)
        
        
        if self.board[x_global][y_global].local_win():
            region_value = 2 if self.x_turn else -2 
            self.board[x_global][y_global].mark_complete(region_value)
            self.game_board[(x_global,y_global)] = move_value
            
        
        self.x_turn = not self.x_turn
    
    
    def get_value(self, x, y):
        x_global = m.floor(x / 3)
        y_global = m.floor(y / 3)
                
        x_local = x % 3
        y_local = y % 3
                
        value = self.board[x_global][y_global].get_c(x_local, y_local)
        return value
        
    def show(self):
        print(self.game_board)
    
    def free_move(self, x, y):
        if (x,y) not in self.taken_squares:
            self.taken_squares.append((x,y))
            return True
        else:
            return False
    
    
    def legal_moves(self):
        
        legal = []
        for x in range(9):
            for y in range(9):
                x_global = m.floor(x / 3)
                y_global = m.floor(y / 3)
                
                x_local = x % 3
                y_local = y % 3
                
                if self.board[x_global][y_global].get_c(x_local, y_local) == 0:
                    legal.append((x,y))
                
        return legal
    
    

        
        
        
        
        
          
# game()