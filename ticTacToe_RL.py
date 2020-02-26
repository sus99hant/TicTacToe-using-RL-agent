from __future__ import print_function, division
from builtins import range, input
import numpy as np
import pandas as pd
LENGTH = 3
def play_game(p1,p2,env,draw = False):
    
    current_player = None
    
    while not env.game_over():
        if current_player == p1:
            current_player = p2
        else :
            current_player = p1
            
        if draw:
            if (draw == 1 and current_player == p1) :
                env.draw_board()
            if (draw == 2 and current_player == p2) :
                env.draw_board()
            
        #current player makes action 
        current_player.take_action(env)
        
        #update histories
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)
        
    if draw:
        env.draw_board()
        
    #we perform the value function update
    p1.update(env)
    p2.update(env)
    
def get_state_hash_and_winner(env, i=0, j=0):
    results = []

    for v in (0, env.x, env.o):
        env.board[i,j] = v # if empty board it should already be 0
        if j == 2:
      # j goes back to 0, increase i, unless i = 2, then we are done
            if i == 2:
        # the board is full, collect results and return
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i + 1, 0)
        else:
      # increment j, i stays the same
            results += get_state_hash_and_winner(env, i, j + 1)

    return results

def initialV_x(env,state_winner_triples):
    V= np.zeros(env.num_states)
    for state, ended, winner in state_winner_triples:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v =0.5
        V[state] = v
    return V  

def initialV_o(env,state_winner_triples):
    V= np.zeros(env.num_states)
    for state, ended, winner in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v =0.5
        V[state] = v
    return V  

class Environment:
  def __init__(self):
    self.board = np.zeros((LENGTH, LENGTH))
    self.x = -1 # represents an x on the board, player 1
    self.o = 1 # represents an o on the board, player 2
    self.winner = None
    self.ended = False
    self.num_states = 3**(LENGTH*LENGTH)

  def is_empty(self, i, j):
    return self.board[i,j] == 0

  def reward(self, sym):
    # no reward until game is over
    if not self.game_over():
      return 0

    # if we get here, game is over
    # sym will be self.x or self.o
    return 1 if self.winner == sym else 0

  def get_state(self):
    # returns the current state, represented as an int
    # from 0...|S|-1, where S = set of all possible states
    # |S| = 3^(BOARD SIZE), since each cell can have 3 possible values - empty, x, o
    # some states are not possible, e.g. all cells are x, but we ignore that detail
    # this is like finding the integer represented by a base-3 number
    k = 0
    h = 0
    for i in range(LENGTH):
      for j in range(LENGTH):
        if self.board[i,j] == 0:
          v = 0
        elif self.board[i,j] == self.x:
          v = 1
        elif self.board[i,j] == self.o:
          v = 2
        h += (3**k) * v
        k += 1
    return h

  def game_over(self, force_recalculate=False):
    # returns true if game over (a player has won or it's a draw)
    # otherwise returns false
    # also sets 'winner' instance variable and 'ended' instance variable
    if not force_recalculate and self.ended:
      return self.ended
    
    # check rows
    for i in range(LENGTH):
      for player in (self.x, self.o):
        if self.board[i].sum() == player*LENGTH:
          self.winner = player
          self.ended = True
          return True

    # check columns
    for j in range(LENGTH):
      for player in (self.x, self.o):
        if self.board[:,j].sum() == player*LENGTH:
          self.winner = player
          self.ended = True
          return True

    # check diagonals
    for player in (self.x, self.o):
      # top-left -> bottom-right diagonal
      if self.board.trace() == player*LENGTH:
        self.winner = player
        self.ended = True
        return True
      # top-right -> bottom-left diagonal
      if np.fliplr(self.board).trace() == player*LENGTH:
        self.winner = player
        self.ended = True
        return True

    # check if draw
    if np.all((self.board == 0) == False):
      # winner stays None
      self.winner = None
      self.ended = True
      return True

    # game is not over
    self.winner = None
    return False

  def is_draw(self):
    return self.ended and self.winner is None

  # Example board
  # -------------
  # | x |   |   |
  # -------------
  # |   |   |   |
  # -------------
  # |   |   | o |
  # -------------
  def draw_board(self):
    for i in range(LENGTH):
      print("-------------")
      for j in range(LENGTH):
        print("  ", end="")
        if self.board[i,j] == self.x:
          print("x ", end="")
        elif self.board[i,j] == self.o:
          print("o ", end="")
        else:
          print("  ", end="")
      print("")
    print("-------------")

class Agent:
    def __init__(self, eps=0.1,alpha =0.5):
        self.eps =eps #propability of chossing random action instead of greedy
        self.alpha =alpha # learning rate
        self.verbose = False
        self.state_history =[]
        
    def setV(self,V):
        self.V =V
        
    def set_symbol(self, sym):
        self.sym=sym
            
    def set_verbose(self,v):
        self.verbose=v
    
    def reset_history(self):
        self.state_history =[]
        
    def take_action(self,env):
        #chosse a random action based epsilon greedy
        r= np.random.rand()
        best_state =None
        next_move =None 
        if r < self.eps:
            #take a random action
            if self.verbose:
                print("take a random action")
            
            possible_moves=[]
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i,j):
                        possible_moves.append((i,j))
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else:
            pos2val={}
            best_value = -1
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i,j):
                        #what is the state after you made this move
                        env.board[i,j] = self.sym
                        state = env.get_state()
                        env.board[i,j] = 0 # reverting back to the original value 
                        pos2val[(i,j)] = self.V[state]
                        if self.V[state] >best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move =(i,j)
                            
            #if verbose true then print the values
            if self.verbose:
                print ("Taking a greedy action")
                for i in range(LENGTH):
                    print ("------------------")
                    for j in range(LENGTH):
                        if env.is_empty(i,j):
                            #print the value
                            print ("%.2f|" % pos2val[(i,j)],end="")
                        else:
                            print (" ",end="")
                            if env.board[i,j] == env.x:
                                print ("x |",end ="")
                            elif env.board[i,j] ==env.o:
                                print ("o |",end = "")
                            else:
                                print (" |",end = "")
                    print("")
                print ("--------------------")
        env.board[next_move[0], next_move[1]] = self.sym

    def update_state_history(self,s):
        self.state_history.append(s)
        
    def update(self,env):
        #we want to backtrack over the states, so that:
        #V[prev_state] = V[prev_state] + alpha *(V[next_state] - V[prev_state])
        reward =env.reward(self.sym)
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha*(target - self.V[prev])
            self.V[prev] = value
            target = value
        self.reset_history()
        
class Human():
    def __init__(self):
        pass
    
    def set_symbol(self,sym):
        self.sym =sym
    
    def take_action(self,env):
        while True:
            move = input("Enter co-ordinates for next move(i,j):")
            i,j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty(i,j):
                env.board[i,j] = self.sym
                break
            
    def update(self,env):
        pass
    
    def update_state_history(self,s):
        pass
    
    
    
if __name__ == '__main__':
    # train the agent 
    p1 = Agent()
    p2 = Agent()
    
    #set initial V for p1 and p2
    env = Environment() 
    state_winner_triples = get_state_hash_and_winner(env)
    
    Vx = initialV_x(env,state_winner_triples)
    p1.setV(Vx)
    Vo = initialV_o(env,state_winner_triples)
    p2.setV(Vo)
    
    # give each player their symbol
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)
    
    T= 25000
    for t in range(T):
        if t%200 ==0:
            print ("EPiSODE:",t)
        play_game(p1,p2,Environment())
    #Human Interface
    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
        play_game(p1,human,Environment(),draw =2)
        #Here agent is player 1 
        answer = input("Play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break

        
