from copy import deepcopy
import math as m
import random as r
import network as cnn
from numba import njit

NULL_PARENT = -10
ROOT = 1
BRANCH = 0
LEAF = -1





class Tree:
    def __init__(self, type, parent, board,route, turn):
        self.type = type
        self.parent = parent
        self.children = []
        self.board = board
        self.route = route
        self.turn = turn
        
        self.wins = 0
        self.explorations = 0
        self.visits = 0
        
        
   
    def expand(self):
        legal_moves = self.board.legal_moves()
        
        if legal_moves == []:
            return False
        
        for move in legal_moves:
            
            child_board = deepcopy(self.board)
            child_board.global_move(move[0], move[1])
            
            child = Tree(LEAF,self,child_board,move, not self.turn)

            self.children.append(child)
            
        r.shuffle(self.children)
            
        if(self.type != ROOT):
            self.type = BRANCH
    
    def update(self,result):
        self.wins += result
        self.explorations += 1
    
    
    
   
    def select_child(self, turn):
        
        # EXP_PARAM = m.sqrt(2)
        EXP_PARAM = 1.0
        DEBUG = False
        
            
            
        def selection_formula(child, parent_exp):
            
            if child.explorations == 0:
                if turn:
                    return m.inf
                else:
                    return -m.inf
                 
            # test = EXP_PARAM * m.sqrt(m.log(parent_exp))
            # print(test)
            
            first_term = child.wins // child.explorations
            second_term = EXP_PARAM * m.sqrt(m.log(parent_exp + 1) / child.explorations)
            
            if turn:
                
                    
                    return first_term + second_term
            else:
                    
                    return first_term - second_term
        
        
        
        favorite_child = self.children[0]
        highest_score = -m.inf
        lowest_score = m.inf
        parent_exp = self.explorations
        
        if DEBUG: print("player turn " +str(turn))
        
        r.shuffle(self.children)
        
        
        for child in self.children:
            
            if turn:
                child_score = selection_formula(child, parent_exp)
                if DEBUG: print("self score " + str(child_score))
                if child_score > highest_score:
                    highest_score = child_score
                    favorite_child = child
            else:
                child_score = selection_formula(child, parent_exp)
                
                if DEBUG: print("other score " + str(child_score))
                if child_score < lowest_score:
                    lowest_score = child_score
                    favorite_child = child
                
        if DEBUG: print("favorite " + str(highest_score))
        if DEBUG: print("favorite " + str(lowest_score))
        return favorite_child
            
        
    def explore(self, model):
        # this is for the neural network
        # the result is the eval for RED
        processed_input = cnn.process_position(self.board, self.turn)
        # print(processed_input)
        prediction = model.forward(processed_input)
        # return r.uniform(0,1)
        
        return prediction.item()
    
    def debug(self):
        
        
        print("route:")
        print(self.route)
        
        
        print("type:")
        print(self.type)
        
        
        
        print("explorations:")
        print(self.explorations)
        
        print("wins:")
        print(self.wins)
        
        print("turn:")
        print(self.turn)
        
        
        print("\n")
        


class MCTS:
    def __init__(self,board, model, turn):
        self.root = Tree(ROOT, None, deepcopy(board), (-1,-1), turn)
        self.model = model
        
        
        
    def select_from_root(self):
        
        DEBUG = False
        turn = True
        node = self.root
        
        
        
        if DEBUG:
            print("\n")
            print("\n")
            print("\n")
            print("START")
        
        while node.type != LEAF:
            
        
            # if node.children == []:
            #     print("NO CHILDREN")
            #     return node
                
            node = node.select_child(turn)
            
            if DEBUG: node.debug()
            
            turn = not turn
            
        if DEBUG: print("STOP")
            
        return node
        
        
    def backpropagate(self,child,result):
        
        DEBUG = False
        child.update(result)
        node = child.parent
        
        # print("PASS 1")
        while node != None:
            
            node = node.parent
            if node != None:
                node.update(result)
    
    def run_MCTS(self, dataset):
        
        
        print("run_MCTS")
        self.root.expand()
        
        node = self.select_from_root()
        # return node.route
        
        DEPTH = 300
        for _iteration in range(DEPTH):
            node = self.select_from_root()
            has_children = node.expand()
            if (has_children == False):
                # print("no child:" + str(_iteration))
                return self.root.select_child(self.root.turn).route
                
            else: 
            
                #should be exploring one of the children
                node = node.select_child(node.turn)
                
                
                result = node.explore(self.model)
                
                #if there is training
                if dataset != None:
                    dataset.append_position(node.board)
                    
                #opponents good nodes are seen as players losses
                if not node.turn:
                    result = 1.0 - result
                    
                self.backpropagate(node,result)
                # print("single move search iteration:")
                # print(_iteration)
            
            
            
            
        
        # self.root.children[0].debug()
        
        
        # for x in self.root.children:
        #     x.debug()
            
            
        best_child = self.root.select_child(self.root.turn)
            
        
        
        
        return best_child.route   
            
        # for x in self.root.children:
        #     x.debug()
            
        