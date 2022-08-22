import pygame
import tictactoe as t
import montecarlo as mcst
import time
import game 
import network as n
import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

def auto_game(model):
    
    DEBUG = False
    red_policy_evals = n.EvalData()
    blue_policy_evals = n.EvalData()
    
    state = t.Ultimate()
    
    model = model.to(torch.float)
    # print(model.dtype())
    running = True
    
    while running:
        # print("GO!")
        first_player = mcst.MCTS(state, model, True)
        first_move = first_player.run_MCTS(red_policy_evals)
        if state.free_move(first_move[0], first_move[1]):
            state.global_move(first_move[0], first_move[1])
        
        # n.process_position(state)
        
        
        red_game_status, red_results = game.game_result(state,True)
        
        if red_game_status:
            red_policy_evals.update_actual(red_results[1])
            red_policy_evals.process_all_positions(True)
            blue_policy_evals.update_actual(1 - red_results[1])
            blue_policy_evals.process_all_positions(False)
            
            complete_data = red_policy_evals.join_data(blue_policy_evals)
            complete_data.randomize_data()
            complete_data.map_tuples()

            if DEBUG:
                print("RED WIN DATA")
                red_policy_evals.show_data()
                print("BLUE LOSS DATA")
                blue_policy_evals.show_data()
                print(red_policy_evals.__len__())
                print(blue_policy_evals.__len__())
            
            return complete_data
        
        
        
        #second player
        
        second_player = mcst.MCTS(state, model, False)
        second_move = second_player.run_MCTS(blue_policy_evals)
        if state.free_move(second_move[0], second_move[1]):
            state.global_move(second_move[0], second_move[1])
            

        
        blue_game_results, blue_results = game.game_result(state,False)
        
        if blue_game_results:
            blue_policy_evals.update_actual(blue_results[1])
            blue_policy_evals.process_all_positions(False)
            red_policy_evals.update_actual(1 - blue_results[1])
            red_policy_evals.process_all_positions(True)
            
            complete_data = red_policy_evals.join_data(blue_policy_evals)
            complete_data.randomize_data()
            complete_data.map_tuples()
            
            
            if DEBUG: 
                print("BLUE WIN DATA")
                blue_policy_evals.show_data()
                print("RED LOSS DATA")
                red_policy_evals.show_data()
                print(blue_policy_evals.__len__())
                print(red_policy_evals.__len__())
            
            return complete_data
        
        
    
        
# auto_game()



def train(dataloader, model, loss_fn, optimizer):
    DEBUG = False
    SHOW = False
    size = len(dataloader.dataset)
    model.train()
    model.to(torch.float)
    
    
    losses = []
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to("cpu"), y.to("cpu")

        if DEBUG: print(X.dtype)
        # Compute prediction error
        pred = model(X[0])
        if DEBUG:
            print(y)
            print(pred)
        loss = loss_fn(pred.to(torch.float), y[0].to(torch.float))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        
        MAX_NORM = 2.0
        torch.nn.utils.clip_grad_norm(model.parameters(), MAX_NORM)
        
        optimizer.step()

        # print(loss.detach().float().item())
        if SHOW: losses.append((batch, loss.detach().float().item()))
        
        # print(f"loss: {loss:>7f}  {batch:>5d}")
        
    if SHOW:
        plt.scatter(*zip(*losses))
        plt.ylabel('some numbers')
        plt.show()
            
            


def continous_train(model, loss_fn, optimizer):
    
    
    for game in range(10000):
        print("game: " + str(game))
        dataset = auto_game(model)
        # dataset.show_data()

        batch_size = 1
        dataloader = DataLoader(dataset, batch_size=batch_size)


        

        train(dataloader, model, loss_fn, optimizer)
        if game % 20 == 0:
            torch.save(model.state_dict(), "cnn.pt")


# continous_train(model, loss_fn, optimizer)




model = n.NeuralNetwork().to("cpu")
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)



model.load_state_dict(torch.load('cnn.pt'))

continous_train(model, loss_fn, optimizer)





# for X, y in dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

# train(dataloader,)