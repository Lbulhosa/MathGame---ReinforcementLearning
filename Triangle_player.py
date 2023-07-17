import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt
import pygame

# Code Mode
speed_mode = 1
train_mode = 1
graph_mode = 0
realgame_mode = 0
ngames = 15000

#colors
BLACK = 0, 0, 0
WHITE = 255, 255, 255
DARK_BROWN = 225, 129, 58
GREEN = 89, 153, 78
BROWN = 243, 198, 119
r_slot = 40
r_piece = 25
pos = [(400,120),(350,200),(450,200),(300,280),(400,280),(500,280),(250,360),(350,360),(450,360),(550,360),(200,440),(300,440),(400,440),(500,440),(600,440)]


# QLearning algorithm
class Q_learning(object):

    def __init__(self, verbose=False, init_fig=False, interactive_plot=False):
        global Qtable
        global state_index, action_index
        state_index = 1
        action_index = 1
        Qtable = np.empty((20000,50),dtype=object)
        Qtable[:] = '0'

    # Print Qtable
    def print_Qtable(self):
        print(Qtable)
        print("\n")

    # Save Qtable to npy file
    def save_Q(self):

        np.save('Q',Qtable,allow_pickle=True)
        return

    # Load Qtable from npy file
    def load_Q(self):
        global state_index,action_index
        global Qtable
        Qtable = np.load('Q.npy',allow_pickle=True)
        print("Qtable was loaded")
        for i in range(1,20000):

            if Qtable[i,0] == '0':
                state_index = i
                break
        for i in range(1,50):

            if Qtable[0,i] == '0':
                action_index = i
                break
            elif i == 50 and Qtable[0,i] != '0':
                action_index = 51
        if(train_mode == 1):
            print("We have ",state_index-1," states")
            print("We have ",action_index-1," actions")
            print(Qtable)
            time.sleep(4)

        return

    # Get current Qvalue for a certain [state,play]
    def get_Qvalue(self,state,play):
        action_pos = 1
        state_pos = 1
        for j in range(1,50):
            if Qtable[0,j] == play:
                action_pos = j
                break
        for i in range(1,20000):
            if Qtable[i,0] == state:
                state_pos = i
                break

        return float(Qtable[i,j])

    # Get the maxQ for a certain state
    def get_maxQ(self,state):
        state_pos = 1
        for i in range(1,20000):
            if Qtable[i,0] == state:
                state_pos = i
                break
        maxQ = 0
        for j in range(1,50):
            if float(Qtable[i,j]) > maxQ:
                maxQ = float(Qtable[i,j])


        return maxQ

    # Update Qvalue for a certain [state,play]
    def update_Qvalue(self,state,play,Qvalue):
        action_pos = 1
        state_pos = 1
        for j in range(1,50):
            if Qtable[0,j] == play:
                action_pos = j
                break
        for i in range(1,20000):
            if Qtable[i,0] == state:
                state_pos = i
                break

        Qtable[i,j] = str(Qvalue)

    # Get the best play for a certain state using the Qtable
    def get_best_play(self,actions,state):
        size = np.size(actions)
        maxQ = 0
        maxQ_action_pos = 0
        state_pos = 1

        for i in range(1,20000):
            if Qtable[i,0] == state:
                state_pos = i
                break

        for i in range(size):
            for j in range(1,50):
                if Qtable[0,j] == actions[i]:
                    if float(Qtable[state_pos,j]) > maxQ:
                        maxQ = float(Qtable[state_pos,j])
                        maxQ_action_pos = j

        if maxQ_action_pos != 0 :
            return Qtable[0,maxQ_action_pos]
        else:
            #print("Q play didn't work. Random play\n")
            r = random.randint(0,size-1)
            return actions[r]

# Environment
class env(object):

    def __init__(self, verbose=False, init_fig=False, interactive_plot=False):
        global board
        global actions
        actions = np.zeros(0)
        board = np.zeros(15)

    # Reset the board
    def reset_board(self):
        for i in range(15):
            board[i] = 1

        r = random.randint(0,5)
        if r == 0: board[0] = 0
        elif r == 1: board[4] = 0
        elif r == 3: board[12] = 0
        else: board[14] = 0




    # Show board (print on terminal)
    def show(self):
        print("    " + str(int(board[0])))
        print("   " + str(int(board[1])) + " " + str(int(board[2])))
        print("  " + str(int(board[3])) + " " + str(int(board[4])) + " " + str(int(board[5])))
        print(" " + str(int(board[6])) + " " + str(int(board[7])) + " " + str(int(board[8])) + " " + str(int(board[9])))
        print("" + str(int(board[10])) + " " + str(int(board[11])) + " " + str(int(board[12])) + " " + str(int(board[13])) + " " + str(int(board[14])))
        print("\n")

    # Check the current state
    def check_state(self):
        state = ''
        for i in range(15):
            if board[i] == 0:
                state = state + str(i)
        return state

    # Check possible actions for the current state
    def check_actions(self):
        global actions
        actions = []


        if board[0] == 1:
            if board[1] == 1 and board[3] == 0:
                actions = np.append(actions,"013")
            if board[2] == 1 and board[5] == 0:
                actions = np.append(actions,"025")

        if board[1] == 1:
            if board[3] == 1 and board[6] == 0:
                actions = np.append(actions,"136")
            if board[4] == 1 and board[8] == 0:
                actions = np.append(actions,"148")

        if board[2] == 1:
            if board[4] == 1 and board[7] == 0:
                actions = np.append(actions,"247")
            if board[5] == 1 and board[9] == 0:
                actions = np.append(actions,"259")

        if board[3] == 1:
            if board[1] == 1 and board[0] == 0:
                actions = np.append(actions,"310")
            if board[4] == 1 and board[5] == 0:
                actions = np.append(actions,"345")
            if board[6] == 1 and board[10] == 0:
                actions = np.append(actions,"3610")
            if board[7] == 1 and board[12] == 0:
                actions = np.append(actions,"3712")

        if board[4] == 1:
            if board[7] == 1 and board[11] == 0:
                actions = np.append(actions,"4711")
            if board[8] == 1 and board[13] == 0:
                actions = np.append(actions,"4813")


        if board[5] == 1:
            if board[2] == 1 and board[0] == 0:
                actions = np.append(actions,"520")
            if board[4] == 1 and board[3] == 0:
                actions = np.append(actions,"543")
            if board[8] == 1 and board[12] == 0:
                actions = np.append(actions,"5812")
            if board[9] == 1 and board[14] == 0:
                actions = np.append(actions,"5914")

        if board[6] == 1:
            if board[3] == 1 and board[1] == 0:
                actions = np.append(actions,"631")
            if board[7] == 1 and board[8] == 0:
                actions = np.append(actions,"678")

        if board[7] == 1:
            if board[4] == 1 and board[2] == 0:
                actions = np.append(actions,"742")
            if board[8] == 1 and board[9] == 0:
                actions = np.append(actions,"789")

        if board[8] == 1:
            if board[4] == 1 and board[1] == 0:
                actions = np.append(actions,"841")
            if board[7] == 1 and board[6] == 0:
                actions = np.append(actions,"876")

        if board[9] == 1:
            if board[8] == 1 and board[7] == 0:
                actions = np.append(actions,"987")
            if board[5] == 1 and board[2] == 0:
                actions = np.append(actions,"952")

        if board[10] == 1:
            if board[6] == 1 and board[3] == 0:
                actions = np.append(actions,"1063")
            if board[11] == 1 and board[12] == 0:
                actions = np.append(actions,"101112")

        if board[11] == 1:
            if board[7] == 1 and board[4] == 0:
                actions = np.append(actions,"1174")
            if board[12] == 1 and board[13] == 0:
                actions = np.append(actions,"111213")

        if board[12] == 1:
            if board[11] == 1 and board[10] == 0:
                actions = np.append(actions,"121110")
            if board[7] == 1 and board[3] == 0:
                actions = np.append(actions,"1273")
            if board[8] == 1 and board[5] == 0:
                actions = np.append(actions,"1285")
            if board[13] == 1 and board[14] == 0:
                actions = np.append(actions,"121314")

        if board[13] == 1:
            if board[8] == 1 and board[4] == 0:
                actions = np.append(actions,"1384")
            if board[12] == 1 and board[11] == 0:
                actions = np.append(actions,"131211")

        if board[14] == 1:
            if board[13] == 1 and board[12] == 0:
                actions = np.append(actions,"141312")
            if board[9] == 1 and board[5] == 0:
                actions = np.append(actions,"1495")


        return actions

    # Apply action to the environment
    def make_play(self, code):
        t = len(code)
        if t == 3:
            first = int(code[0])
            second = int(code[1])
            third = int(code[2])
        elif t == 4:
            for i in range(4):
                if code[i] == '1' and i == 0:
                    aux = code[0]+code[1]
                    first = int(aux)
                    second = int(code[2])
                    third = int(code[3])
                    break
                else:
                    aux = code[2]+code[3]
                    first = int(code[0])
                    second = int(code[1])
                    third = int(aux)
                    break

        else:
            aux = code[0]+code[1]
            first = int(aux)
            aux = code[2]+code[3]
            second = int(aux)
            aux = code[4]+code[5]
            third = int(aux)



        board[first] = 0
        board[second] = 0
        board[third] = 1

        return

    # Calculate reward for the action that was applied
    def get_reward(self):
        count_pieces = 0
        for i in range(15):
            if board[i] == 1:
                count_pieces = count_pieces + 1
        if count_pieces == 1:
            return 1
        else:
            return 0

    # Return number of pieces on the board
    def get_pieces(self):
        count_pieces = 0
        for i in range(15):
            if board[i] == 1:
                count_pieces = count_pieces + 1

        return count_pieces

    # Check state and save it on the Qtable if it's new
    def save_state(self):
        global state_index
        state = ''
        for i in range(15):
            if board[i] == 0:
                state = state + str(i)

        flag = 0
        #print("STATE:", state)

        for i in range(state_index):
            if Qtable[i,0] == state:
                #print("STATE ALREADY EXISTS\n")
                #time.sleep(1)
                flag = 1
                break
        if flag == 0:
            Qtable[state_index,0] = state
            state_index = state_index + 1


        return state

    # Check action and save it on the Qtable if it's new
    def save_action(self,play):
        global action_index

        #print("ACTION:", play)
        flag = 0
        for i in range(1,action_index):
            if Qtable[0,i] == play:
                #print("PLAY ALREADY EXISTS\n")
                #time.sleep(1)
                flag = 1
                break
        if flag == 0:
            Qtable[0,action_index] = play
            action_index = action_index + 1


        return


def main():
    if(realgame_mode == 1):
        pygame.init()
        screen = pygame.display.set_mode((800,600))
        pygame.display.set_caption("Peg Solitaire Board")

##### simulation values

    game = 0
    if(train_mode == 1):
        epsilon = 1
    else:
        epsilon = -1
    alfa = 0.75
    dfactor = 1

##### Extra values
    learning = np.zeros((ngames,2))
    for i in range(ngames):
        learning[i,0] = i
    plotx = []
    ploty = []

##### Init
    jogo = env()
    Q = Q_learning()


#### LOAD Q table
    if(train_mode == 0):
        Q.load_Q()



#### start new game
    jogo.reset_board()
    sucess_games = 0


############## MAIN GAME CODE
    while(1):

####### Check possible actions for the current state
        if(realgame_mode == 0):
            jogo.show()
        elif(realgame_mode == 1):
            screen.fill(BLACK)
            pygame.draw.rect(screen,(0,0,0),pygame.Rect(50,0,700,600))
            for i in range(15):
                pygame.draw.circle(screen,DARK_BROWN,pos[i],r_slot)

            for i in range(15):
                if board[i] == 1:
                    pygame.draw.circle(screen,BROWN,pos[i],r_piece)

            pygame.display.update()
        current_state = jogo.check_state()
        #print(current_state)
        actions = jogo.check_actions()
        size = np.size(actions)

######## Check if game ended
        if size == 0:
            print("Game Over\n")

############ Save data and plot
            #save game info for learning plot
            learning[game,1] = jogo.get_pieces()
            if learning[game,1] == 1:
                sucess_games = sucess_games+1
            print("it:", game ,"Number of pieces:",learning[game,1])

            if(train_mode == 1 and graph_mode == 1):
               plotx = np.append(plotx,game)
               ploty = np.append(ploty,learning[game,1])
               plt.axis([game-50,game,0,15])
               plt.plot(plotx[game-50:game],ploty[game-50:game],color='blue')
               plt.pause(0.00000000000000000000000001)

            game = game + 1
            if(speed_mode == 0):
                time.sleep(1)
            if(game == ngames):break



########### Training epsilon update
            if(train_mode == 1):
                if game == 0.5* ngames:
                    epsilon = 0.5
                elif game == 0.75 * ngames:
                    epsilon = 0.2
                elif game == 0.9 * ngames:
                    epsilon = -1


############restart game
            jogo.reset_board()
            if(realgame_mode == 0):
                jogo.show()
            elif(realgame_mode == 1):
                screen.fill(BLACK)
                pygame.draw.rect(screen,(0,0,0),pygame.Rect(50,0,700,600))
                for i in range(15):
                    pygame.draw.circle(screen,DARK_BROWN,pos[i],r_slot)

                for i in range(15):
                    if board[i] == 1:
                        pygame.draw.circle(screen,BROWN,pos[i],r_piece)

                pygame.display.update()
            current_state = jogo.check_state()
            actions = jogo.check_actions()
            size = np.size(actions)

########Choose ACTION (might be using the Qtable or random)
        if(speed_mode == 0):
            time.sleep(1)

        if random.uniform(0, 1) < epsilon:
            #Explore: select a random action
            if(train_mode == 1):
                print("Random value play\n")
            r = random.randint(0,size-1)
            play = actions[r]

        else:
            #Exploit: select the action with max value (future reward)
            if(train_mode == 1):
                print("Q value play\n")
            play = Q.get_best_play(actions,current_state)

#########apply action to the env and get reward
        jogo.make_play(play)
        reward = jogo.get_reward()

#########check the new board state
        new_state = jogo.save_state()
        jogo.save_action(play)

######### Update Q values
        Qvalue = Q.get_Qvalue(current_state,play)
        max_futureQ = Q.get_maxQ(new_state)
        new_Qvalue = Qvalue + alfa*(reward+dfactor * max_futureQ - Qvalue)
        Q.update_Qvalue(current_state,play,new_Qvalue)



##### Save Q table
    Q.save_Q()
    np.savetxt('learning.csv',learning)
    print("Qtable was saved and game experience is over\n")
    #print(Qtable)
    print("Games concluded with sucess: ",sucess_games)

main()
