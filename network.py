import numpy as np 
import torch
from torch import nn
from torch.utils.data import Dataset
import random as r



def process_position(state, turn):
    unified_position = np.zeros((9,9))
    for x in range(9):
            for y in range(9):
                sq = state.get_value(x,y)
                unified_position[x][y] = sq
    
    RED_TURN_MARKER = torch.ones((1,9,9))
    BLUE_TURN_MARKER = torch.zeros((1,9,9))
    
    # print(torch.from_numpy(unified_position))
    tensor = torch.from_numpy(unified_position)
    tensor = tensor.expand(1, tensor.shape[0],tensor.shape[1])
    # print(tensor.shape)
    
    if turn: 
        tensor = torch.stack((tensor, RED_TURN_MARKER), dim = 1)
        # print(tensor.shape)
    else:
        tensor = torch.stack((tensor, BLUE_TURN_MARKER), dim = 1)
        # print(tensor.shape)
        
    # print(tensor.to(torch.float).dtype)
    return tensor.to(torch.float)

class EvalData(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.policy_on_positions = []
        
    def __len__(self):
        return len(self.policy_on_positions)
    
    def __getitem__(self,index):
        return (self.policy_on_positions[index][0], self.policy_on_positions[index][1])
    
    def append_position(self, position_tensor):
        # position prediction actual
        self.policy_on_positions.append([position_tensor, None])
        
    def process_all_positions(self, turn):
        for position in self.policy_on_positions:
            position[0] = process_position(position[0], turn)
            # print(position)
        
    def update_actual(self, result):
        for position in self.policy_on_positions:
            position[1] = result
            
            
    def show_data(self):
        for position in self.policy_on_positions:
            print(position)
            
    def join_data(self, to_join):
        self.policy_on_positions = self.policy_on_positions + to_join.policy_on_positions
        return self
        
    def randomize_data(self):
        r.shuffle(self.policy_on_positions)
        
    def map_tuples(self):
        tuples_position = []
        for position in self.policy_on_positions:
            position = (position[0], position[1])
            tuples_position.append(position)
        self.policy_on_positions = tuples_position
        
        
            
            
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # initialize first set of CONV => RELU => POOL layers
        self.net = nn.Sequential(
        nn.Conv2d(in_channels = 2, out_channels = 16, kernel_size = (6,6)), 
        nn.ReLU(),
        nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3)), 
        nn.ReLU(),
        nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = (2,2)), 
        nn.ReLU(),
        # nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (1,1), 
        # )
        )
        
        # self.net = nn.Sequential(
        # nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3)), 
        # nn.ReLU(),
        # # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # )

    def forward(self,x):
        
        # for layer in self.net:
        #     x = layer(x)
        #     print(x.size())
        
        
        # print(x)
        
        # print(x.size())
        self.to(torch.float)
        prediction = self.net(x)
        # print("prediction:")
        # print(prediction.item())
        prediction = torch.flatten(prediction)
        # print(prediction)
        return prediction.to(torch.float)
    
device = "cpu"
model = NeuralNetwork().to(device)







# x = torch.randn(1, 2, 9, 9)
# model.forward(x)
