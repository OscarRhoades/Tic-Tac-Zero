import pygame
import tictactoe as t
import montecarlo as mcst
import time
import network as n


import torch
from torch import nn

pygame.init()

HEIGHT = 990
WIDTH = 990
BLACK = (0,0,0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
# Set up the drawing window


MARGIN = 5



CUBE_HEIGHT = (HEIGHT - (MARGIN * 10) )/ 9
CUBE_WIDTH = (WIDTH - (MARGIN * 10) )/ 9

def draw_rect(screen, color, x, y):
    pygame.draw.rect(screen, color, [(MARGIN + CUBE_WIDTH) * y + MARGIN,
                              (MARGIN + CUBE_HEIGHT) * x + MARGIN,
                              CUBE_WIDTH,
                              CUBE_HEIGHT])
    
def render(screen, game_state):
    for x in range(9):
            for y in range(9):
                square = game_state.get_value(x,y)
                
                LIGHT_RED = (255, 120, 120)
                LIGHT_BLUE = (120, 120, 255)
                
                
                if square == 1:
                    draw_rect(screen,RED,x,y)
                elif square == -1:
                    draw_rect(screen,BLUE,x,y)
                elif square == 2:
                    draw_rect(screen,LIGHT_RED,x,y)
                elif square == -2:
                    draw_rect(screen,LIGHT_BLUE,x,y)
                else:
                    draw_rect(screen,WHITE,x,y)
                    
def player_move(state, x, y):
    if state.free_move(x, y):
            state.global_move(x, y)
            
def game_result(game_state, turn):
    # checks if there is a win for that player, and then gives the score of the player and 
    message = "VICTORY FOR RED PLAYER" if turn else "VICTORY FOR BLUE PLAYER"
    
    result = game_state.win()
    if result:
        print(message)
        # pygame.time.wait(10000)
        
        
        captures = 0
        for row in game_state.game_board:
            for col in row:
                if col == 1:
                    captures += 1
                elif col == -1:
                    if col == -1:
                        captures -= 1
                        
        game_statistics = (turn, ((captures if captures > 0 else abs(captures)) + 6) / 11)
        print(game_statistics)
               
                  
        return (1, game_statistics)
    
    
    elif game_state.draw():
        print("DRAW")
        # pygame.time.wait(10000)
        
        
        captures = 0
        for row in game_state.game_board:
            for col in row:
                
                if col == 1:
                    captures += 1
                elif col == -1:
                    if col == -1:
                        captures -= 1
        
        #the score is the differential in game board results
        game_statistics = (None, captures if captures > 0 else abs(captures))
        print(game_statistics)
        
        return (-1, game_statistics)
    else:
        return (0, 0)  

        
def self_game():
        
    DISPLAY = True
    state = t.Ultimate()
    
    
    

    if(DISPLAY):
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode([HEIGHT, WIDTH])

    running = True
    while running:
        
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                first_player = mcst.MCTS(state)
                first_move = first_player.run_MCTS()
                if state.free_move(first_move[0], first_move[1]):
                    state.global_move(first_move[0], first_move[1])
                    pygame.time.wait(100)
                        
                # if state.win():
                #     print("VICTORY FOR RED PLAYER")
                #     pygame.time.wait(10000)
                #     return 0
                # elif state.draw():
                #     print("DRAW")
                #     pygame.time.wait(10000)
                #     return 0
                
                if game_result(state,True):
                    return 0
                
                
                #second player
                
                second_player = mcst.MCTS(state)
                second_move = second_player.run_MCTS()
                if state.free_move(second_move[0], second_move[1]):
                    state.global_move(second_move[0], second_move[1])
                    pygame.time.wait(100)
                    
                    
                # if(state.win()):
                #     print("VICTORY FOR BLUE PLAYER")
                #     pygame.time.wait(10000)
                #     return 0
                # elif state.draw():
                #     print("DRAW")
                #     pygame.time.wait(10000)
                #     return 0
                
                if game_result(state,False):
                    return 0
                    
                
                # print("Click ", pos, "Grid coordinates: ", row, column)
                if(DISPLAY):
                    render(screen, state)
                
                # if(state.win()):
                #     print("VICTORY")
                    # exit(0)
                    
        
        if DISPLAY:
            screen.fill(BLACK)
            clock.tick(30)
            
            render(screen,state)
            

            # Flip the display
            pygame.display.flip()      
        
                     
def game():
    
    model = n.NeuralNetwork().to("cpu")
    # model.load_state_dict(torch.load('cnn.pt'))
    state = t.Ultimate()
    
    
    clock = pygame.time.Clock()
    
    
    
    screen = pygame.display.set_mode([HEIGHT, WIDTH])

    running = True
    while running:
        
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                column = pos[0] // (CUBE_WIDTH + MARGIN)
                row = pos[1] // (CUBE_HEIGHT + MARGIN)
                if state.free_move(row, column):
                    state.global_move(row, column)
                
                    if(state.win()):
                        print("VICTORY")
                        time.sleep(10)
                        return 0
                        
                    clock.tick(100)
                    tree = mcst.MCTS(state, model, False)
                    tree_move = tree.run_MCTS(None)
                    if state.free_move(tree_move[0], tree_move[1]):
                        state.global_move(tree_move[0], tree_move[1])
                
                    if(state.win()):
                        
                        print("VICTORY")
                        time.sleep(10)
                        return 0
                    
                
                # print("Click ", pos, "Grid coordinates: ", row, column)
               
                render(screen, state)
                
                # if(state.win()):
                #     print("VICTORY")
                    # exit(0)
                    
        
        screen.fill(BLACK)
        clock.tick(30)
        
        render(screen,state)
        

        # Flip the display
        pygame.display.flip()

# Done! Time to quit.
# pygame.quit()

# auto_game()
# self_game()
game()