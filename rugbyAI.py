import copy
import time
import datetime
import random
from random import randint as rng
from itertools import permutations

print('hello importing',datetime.datetime.now())
# import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
print('ok imported',datetime.datetime.now())


screenwidth,screenheight = (1000,600)
playerwidth,playerheight=(50,50)
oppovel = 8
walllist=list()

shutdown=False
win=None
e=0


#Pygame Stuff
import pygame
pygame.init()
win=pygame.display.set_mode((screenwidth,screenheight))
pygame.display.set_caption('Wobble wobble feeder')


def overlap(pos1,pos2):
    #pos = x,y,width,height
    left  = max(pos1[0], pos2[0])
    top   = max(pos1[1], pos2[1])
    right = min(pos1[0]+pos1[2], pos2[0]+pos2[2])
    bot   = min(pos1[1]+pos1[3], pos2[1]+pos2[3])
    if right-left > 0 and bot-top > 0:
        area = (left-right) * (top-bot)
    else:
        area = 0
    return area


class Player():
    def __init__(self,x=screenwidth//2-playerwidth//2,
                        y=screenheight-50,
                        width=playerwidth,
                        height=playerheight,
                        xvel=8,yvel=8):
        self.x=x
        self.y=y         
        self.width=width
        self.height=height
        self.xvel=xvel
        self.yvel=yvel
        self.pos=(self.x, self.y, self.width, self.height)
        self.reward=0
        self.action=0
        self.ball=False
    
    def movement(self):
        if self.action==0:  # move left
            if self.x>0:
                self.x-=self.xvel
                # self.reward-=0.1
        elif self.action==1: #move up
            self.y-=self.yvel
            if self.y <-1: self.y=-1
            
        elif self.action==2:  # move right
            if self.x+self.width<screenwidth:
                self.x+=self.xvel
                # self.reward-=0.1

        self.updatepos()

    def updatepos(self):
        self.pos = self.x, self.y, self.width, self.height


class Opponent(Player):
    def __init__(self,x=screenwidth//2-playerwidth//2,
                        y=screenheight//2,
                        width=playerwidth+20,
                        height=playerheight+20,
                        xvel=oppovel,yvel=oppovel):
        super().__init__(x,y,width,height,xvel,yvel)
        self.action=-1
    

    
class Wall():
    def __init__(self,x,y,width,height):
        self.x=x
        self.y=y         
        self.width=width
        self.height=height
        self.pos=(self.x, self.y, self.width, self.height)
        walllist.append(self)


Wall(-5,0,5,screenheight)  # Left wall
Wall(screenwidth,0,5,screenheight)  # Right wall
# Wall(0,-5,screenwidth,5)

class Env():
    def __init__(self):
        self.player1=Player(x=screenwidth//4-playerwidth//2, y = screenheight-50)
        self.player2=Player(x=screenwidth//4*3-playerwidth//2, y = screenheight-50)
        self.player3=Player(x=screenwidth//2-playerwidth//2, y = screenheight-50-2*playerheight)
        #self.player1.ball=True
        random.choice((self.player1,self.player2,self.player3)).ball=True
        if e<100:
            self.oppo1=Opponent(x=10*(e-1), y = 0)
        else:
            self.oppo1=Opponent(x=screenwidth//2-playerwidth//2, y = 0)
        self.count=0
        self.done=False

    def reset(self):
        del self.player1, self.player2, self.player3, self.oppo1
        self.__init__() 
        state = tuple(item*0.001 for item in [self.player1.x, self.player1.y, self.player2.x, self.player2.y,self.player3.x,self.player3.y, self.oppo1.x, self.oppo1.y,100 if self.player1.ball else 200 if self.player2.ball else 300]) # State after all actions taken
        return state


    def oppomovement(self):
        playerlist=[self.player1,self.player2,self.player3]
        for player in playerlist:  #Oppo movements
            if player.ball:
                if player.x>=self.oppo1.x+playerwidth:
                    self.oppo1.x+=self.oppo1.xvel
                elif player.x+playerwidth<=self.oppo1.x:
                    self.oppo1.x-=self.oppo1.xvel
                if player.y>=self.oppo1.y+playerheight:
                    self.oppo1.y+=self.oppo1.yvel
                elif player.y+playerheight<=self.oppo1.y:
                    self.oppo1.y-=self.oppo1.yvel
                break
        self.oppo1.updatepos() 
    
    def playermovement(self):
        playerlist=[self.player1,self.player2,self.player3]
        for player in playerlist:
            player.reward=0
            player.movement()

        #Pass the ball
        perms = list(permutations(playerlist, 2))
        for p1, p2 in perms:
            if p1.action==4 and p1.ball: #3 players
                p1.action=3
                continue
            elif p1.action==3 and p1.ball:
                p1.ball, p2.ball=False,True
                m=(p2.y-p1.y)/(p2.x -p1.x)
                c=p2.y - p2.x*m
                y1, y2 = m*self.oppo1.x + c, m*(self.oppo1.x+playerwidth) + c 
                if (self.oppo1.y<y1<self.oppo1.y+playerheight or 
                    self.oppo1.y<y2<self.oppo1.y+playerheight or 
                    y1<self.oppo1.y<y2):
                    self.done=True
                    p1.reward-=10
                break

        if e>100: self.oppomovement()

        for player in playerlist:  # - reward if defender hit player
            if overlap(self.oppo1.pos, player.pos):
                if player.ball:
                    self.done=True
                player.reward-=10 

        #Win check
        # playerlist.sort(key=lambda player: player.y)
        for player in playerlist:
            if player.ball:
                ballpos=player.y
        if ballpos<=0: 
            self.done=True
            for player in playerlist: 
                player.reward+=10
            
        if self.done: 
            weighted_pos = min(10, (screenheight-ballpos)/60)
            for player in playerlist:
                player.reward += weighted_pos
            if e%10==0: print('pos:{:.3}'.format(float(weighted_pos)),end=', ')
               

    def runframe(self, action1,action2, action3):
        self.done=False
        self.player1.action=action1
        self.player2.action=action2
        self.player3.action=action3
        self.playermovement()

        state = tuple(item*0.001 for item in [self.player1.x, self.player1.y, self.player2.x, self.player2.y,self.player3.x,self.player3.y, self.oppo1.x, self.oppo1.y,100 if self.player1.ball else 200 if self.player2.ball else 300])  # State after all actions taken

        return (state, (self.player1.reward, self.player2.reward,self.player3.reward), self.done)

    def render(self):
        if win is not None:
            # pygame.event.get()
            #time.sleep(0.04)
            win.fill((0,0,0))
            pygame.draw.rect(win,  (255,255,255) if self.player1.ball else (255,0,0), self.player1.pos)  #Red
            pygame.draw.rect(win,  (255,255,255) if self.player2.ball else (255,255,0), self.player2.pos)  #Yellow
            pygame.draw.rect(win,  (255,255,255) if self.player3.ball else (0,255,0), self.player3.pos)  #Green
            pygame.draw.rect(win, (0,0,255), self.oppo1.pos)  #Blue
            pygame.display.update()


env = Env()
state_size = 9 #env.observation_space.shape[0]
action_size = 5 #env.action_space.n
batch_size = 64
n_episodes = 1000
output_dir = 'data/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DQNAgent:
    def __init__(self,state_size,action_size,num=1):
        self.num=num
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95
        self.epsilon = 0.0 #exploitation=0 vs exploration=1
        self.epsilon_decay = 0.999 #less and less each time
        self.epsilon_min = 0.01 #1% exploration
        self.learning_rate = 0.001
        self.model = self._build_model()

        
    
    def _build_model(self):
        model=Sequential()
        model.add(Dense(64,input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
    
        return model
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
        
    def act(self,state):
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
        
    def replay(self,batch_size):
        minibatch=random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minibatch:
            target=reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action]=target
            self.model.fit(state,target_f,epochs=1,verbose=0)
            
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

    def load(self,name):
        self.model.load_weights(name) 
        print('loaded',name)


    def save(self,name):
        self.model.save_weights(name)



agent1 = DQNAgent(state_size,action_size)
agent2 = DQNAgent(state_size,action_size)
agent3 = DQNAgent(state_size,action_size)
agentlist = [agent1,agent2,agent3]

count=0
maxscore=0

#Loading previous versions
num = 5678
if num:
    [agent.load(output_dir + 'agent{}_{:05d}.hdf5'.format(i, num)) for i, agent in enumerate(agentlist,1)]

print('Starting loop')
e=300
#for e in range(n_episodes):
while True:
    if shutdown: break
    e+=1
    state=env.reset() 
    for t in range(5000): #Gameloop
        if shutdown: break
        state1 = state
        state2 = state[2:4]+state[:2]+state[4:]
        state3 = state[4:6]+state[:4]+state[6:]

        state1=np.reshape(state1,[1, state_size])
        state2=np.reshape(state2,[1, state_size])
        state3=np.reshape(state3,[1, state_size])


        if e<100:
            action1, action2, action3=1,1,1
        else:
            action1 = agent1.act(state1)
            action2 = agent2.act(state2)
            action3 = agent3.act(state3)

 

        if win is not None:
            events = pygame.event.get()
            for event in events: 
                if event.type==pygame.QUIT:
                    shutdown=True
                    break

            keys = pygame.key.get_pressed()          #player 1
  
            if keys[pygame.K_a]:
                action1=0
            elif keys[pygame.K_w]:
                action1=1
            elif keys[pygame.K_d]:
                action1=2
            elif keys[pygame.K_n]:
                action1=3
            elif keys[pygame.K_m]:
                action1=4

            env.render()
        

        next0, rewards, done = env.runframe(action1,action2,action3) #next0 means next_state

        next1 = next0
        next2 = next0[2:4]+next0[:2]+next0[4:]
        next3 = next0[4:6]+next0[:4]+next0[6:]


        next1=np.reshape(next1,[1, state_size])
        next2=np.reshape(next2,[1, state_size])
        next3=np.reshape(next3,[1, state_size])

        agent1.remember(state1, action1, rewards[0], next1, done)
        agent2.remember(state2, action2, rewards[1], next2, done)
        agent3.remember(state3, action3, rewards[2], next3, done)

        state=next0

        if done:
            break


    

print('saved best file')

#import shutil 
#shutil.rmtree('data/')


