import pygame, sys
from pygame.locals import *
from random import *
import numpy as np # q learning bit
import time, copy

AI = False
VISUAL_GAME = True

#import os
#os.environ["SDL_VIDEODRIVER"] = "dummy"

if VISUAL_GAME:
    pygame.init()
    clock = pygame.time.Clock()

#FPS = 60
#fpsClock = pygame.time.Clock()

# set up window
WIDTH = 40
HEIGHT = 30
reward = False
done = False
prev_state = None
prev_action = None

if VISUAL_GAME:
    DISPLAYSURF = pygame.display.set_mode((10 * WIDTH, 10 * HEIGHT), 0, 32)
    pygame.display.set_caption('Shane Snake Game - Score: 0')
else:
    DISPLAYSURF = []
    for _ in range(HEIGHT):
        current_row = []
        for _ in range(WIDTH):
            current_row.append("X")
        DISPLAYSURF.append(current_row)
    #for s in DISPLAYSURF:
        #print(*s)    

    

displayWidth = WIDTH * 10
displayHeight = HEIGHT * 20 / 2

#catImg = pygame.image.load('cat.png')
#catx = 10
#caty = 10

# set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0 , 255)

SPEED = 100 # lower number => faster snake

score = 0
gameOver = False


# draw on surface object
#pygame.draw.line(DISPLAYSURF, BLUE, (240, 240), (120, 60), 4)
#pygame.draw.polygon(DISPLAYSURF, GREEN, ((146, 0), (291, 106), (236, 277), (56, 277), (0, 106)))

if VISUAL_GAME:
    DISPLAYSURF.fill(WHITE) 

#pygame.Rect(100, 100, 10, 10
if VISUAL_GAME:
    snakeRect = Rect(100, 100, 10, 10)
else:
    snakeRect = {"x": 20, "y": 10}
    DISPLAYSURF[snakeRect["y"]][snakeRect["x"]] = "O"
    
listSnakeSquares = [snakeRect]
snakeHead = listSnakeSquares[0]

foodRect = None

direction = None
timeSinceLastAction = 0


def print_display_surf(episode_num=None):
    global DISPLAYSURF
    assert(not VISUAL_GAME)
    DISPLAYSURF = []
    for _ in range(HEIGHT):
        current_row = []
        for _ in range(WIDTH):
            current_row.append("'")
        DISPLAYSURF.append(current_row)
    
    for snakePart in listSnakeSquares:
        DISPLAYSURF[snakePart["y"]][snakePart["x"]] = "O"
    
    if foodRect:
        DISPLAYSURF[foodRect["y"]][foodRect["x"]] = "$"
    for snakePart in listSnakeSquares:
        DISPLAYSURF[snakePart["y"]][snakePart["x"]] = "O"    
        
#    for square in DISPLAYSURF:
#        print(square)
        
    print(f"================SCORE: {score}==================")
    for i in DISPLAYSURF:
        print(''.join(i))
    if episode_num:
        print(f"EPISODE: {episode_num}")
    #print(DISPLAYSURF)
    print("============================================")

def text_to_visual(textBoard, first_time, score):
    #global DISPLAYSURF
    if first_time:
        pygame.init()
        clock = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((10 * WIDTH, 10 * HEIGHT), 0, 32)
    pygame.display.set_caption('Shane Snake Game - Score: {}'.format(score))
    DISPLAYSURF.fill(WHITE)
    for index_height, y in enumerate(textBoard):
        for index_width, x in enumerate(y):
            if x == "O":
                snakePart = Rect(index_width * 10, index_height * 10, 10, 10)
                pygame.draw.rect(DISPLAYSURF, BLUE, snakePart, 0)
                pygame.draw.rect(DISPLAYSURF, BLACK, snakePart, 1)                    
            elif x == "$":
                foodRect = Rect(index_width * 10, index_height * 10, 10, 10)
                pygame.draw.rect(DISPLAYSURF, GREEN, foodRect, 0)
                pygame.draw.rect(DISPLAYSURF, BLACK, foodRect, 1)                                    
    pygame.display.update()


def action(collide):
    #print('==============DIRECTION====================')
    #print(direction)
    global foodRect
    newSnakePart = None
    global score
    if collide:
        score += 10
        if VISUAL_GAME:
            pygame.display.set_caption('Shane Snake Game - Score: {}'.format(score))
        foodRect = None
        snakeEnd = listSnakeSquares[len(listSnakeSquares) - 1]
        newSnakePart = snakeEnd.copy()
    if VISUAL_GAME:
        prevPartX = snakeHead.x
        prevPartY = snakeHead.y
    else:
        prevPartX = snakeHead["x"]
        prevPartY = snakeHead["y"]      
    for index, snakePart in enumerate(listSnakeSquares):
        if index:
            if VISUAL_GAME:
                tempX = snakePart.x 
                tempY = snakePart.y 
                snakePart.x = prevPartX
                snakePart.y = prevPartY
            else:
                tempX = snakePart["x"] 
                tempY = snakePart["y"]
                snakePart["x"] = prevPartX
                snakePart["y"] = prevPartY                
            prevPartX = tempX
            prevPartY = tempY
    
    if direction == 'left':
        if VISUAL_GAME:
            snakeHead.move_ip(-10,0)
        else:
            snakeHead["x"] = snakeHead["x"] - 1
    elif direction == 'right':
        if VISUAL_GAME:
            snakeHead.move_ip(10,0)
        else:
            snakeHead["x"] = snakeHead["x"] + 1         
    elif direction == 'up':
        if VISUAL_GAME:
            snakeHead.move_ip(0,-10)
        else:
            snakeHead["y"] = snakeHead["y"] - 1           
    elif direction == 'down':
        if VISUAL_GAME:
            snakeHead.move_ip(0,10)
        else:
            snakeHead["y"] = snakeHead["y"] + 1    
    
    if newSnakePart:
        listSnakeSquares.append(newSnakePart)
    
    for snakePart in listSnakeSquares:
        if VISUAL_GAME:
            pygame.draw.rect(DISPLAYSURF, BLUE, snakePart, 0)
            pygame.draw.rect(DISPLAYSURF, BLACK, snakePart, 1)
        else:
            try:
                DISPLAYSURF[snakePart["y"]][snakePart["x"]] = "O"
            except:
                break

def isDead():
    endGame = False
    if VISUAL_GAME:
        if snakeHead.collidelist(listSnakeSquares[1:]) != -1:
            endGame = True
        if snakeHead.x < 0 or snakeHead.y < 0 or snakeHead.x >= WIDTH * 10 or snakeHead.y >= HEIGHT * 10:
            endGame = True
    else:
        for snakePart in listSnakeSquares[1:]:
            if snakeHead == snakePart:
                endGame = True
        if snakeHead["x"] < 0 or snakeHead["y"] < 0 or snakeHead["x"] >= WIDTH or snakeHead["y"] >= HEIGHT:
            endGame = True
    
    return endGame
        

def changeFood():
    foundFoodSpot = False
    global foodRect  
    while not foundFoodSpot:
        #print('IN HERE')
        x = randint(1, WIDTH-1)
        y = randint(1, HEIGHT-1)
        if VISUAL_GAME:
            foodRect = Rect(x * 10, y * 10, 10, 10)
            foundFoodSpot = True if foodRect.collidelist(listSnakeSquares) == -1 else False
        else:
            foodRect = {"x": x, "y": y}
            foundFoodSpot = True if foodRect not in listSnakeSquares else False
        #print(foundFoodSpot)
    

def TextObjects(text, font):
    assert(VISUAL_GAME)
    textSurface = font.render(text, True, BLACK)
    return textSurface, textSurface.get_rect()

def FinalText(text):
    if VISUAL_GAME:
        largeText = pygame.font.Font('freesansbold.ttf',20)
        TextSurf, TextRect = TextObjects(text, largeText)
        TextRect.center = ((displayWidth/2),(displayHeight/2))
        DISPLAYSURF.blit(TextSurf, TextRect)
    else:
        print('========================================')
        print('=========       GAME OVER      =========')
        print('========================================')

    #pygame.display.update()

    #time.sleep(2)

def reset(render=True):
    global listSnakeSquares
    if VISUAL_GAME:
        pygame.display.set_caption('Shane Snake Game - Score: 0')
    global score, gameOver, snakeRect, listSnakeSquares, snakeHead, foodRect, direction, timeSinceLastAction, DISPLAYSURF
    
    score = 0
    gameOver = False
    if VISUAL_GAME:
        DISPLAYSURF.fill(WHITE) 
        snakeRect = Rect(100, 100, 10, 10)
    else:
        snakeRect = {"x": 20, "y": 10}
        DISPLAYSURF = []
    
    #pygame.Rect(100, 100, 10, 10)
    listSnakeSquares = [snakeRect]
    snakeHead = listSnakeSquares[0]
    
    foodRect = None
    
    direction = None
    timeSinceLastAction = 0
    if not VISUAL_GAME and render:
        print_display_surf()

###################################QL
def get_states(): # QL
    # 0 -> 390, 0 -> 290
    try:
        x_pos_head_to_food = snakeHead.x - foodRect.x # 1 [0 -> 39]
        y_pos_head_to_food = snakeHead.y - foodRect.y # 2 [0 -> 29]
        
        snakeTail = listSnakeSquares[-1]
        x_pos_head_to_tail = snakeHead.x - snakeTail.x # 3 [0 -> 39]
        y_pos_head_to_tail = snakeHead.y - snakeTail.y # 4 [0 -> 29]
        len_of_snake = len(listSnakeSquares) # 5 [0 -> 100 {1200}]
        
        pos_to_top_wall = snakeHead.y # 6 [0 -> 29]
        pos_to_bottom_wall = 290 - snakeHead.y # 7 [0 -> 29]
        #print(pos_to_top_wall, pos_to_bottom_wall)
        pos_to_right_wall = snakeHead.x # 8 [0 -> 39]
        pos_to_left_wall = 390 - snakeHead.x # 9 [0 -> 39]
        #print(pos_to_right_wall, pos_to_left_wall) 
        return [x_pos_head_to_food, y_pos_head_to_food,
                x_pos_head_to_tail, y_pos_head_to_tail,
                len_of_snake, pos_to_top_wall,
                pos_to_bottom_wall, pos_to_right_wall,
                pos_to_left_wall] # len_of_snake
    except:
        return [-1] * 9
    
    #print(x_pos_head_to_food)
    #print(y_pos_head_to_food)
    #print(x_pos_head_to_tail, y_pos_head_to_tail)


def get_states_easier(): # QL
    if foodRect:
        food_right = 1 if snakeHead["x"] < foodRect["x"] else 0 if snakeHead["x"] == foodRect["x"] else 2 #1 [3] [2; 0; 1]
        food_below = 1 if snakeHead["y"] < foodRect["y"] else 0 if snakeHead["y"] == foodRect["y"] else 2 #2 [3] [2; 0; 1]
    else:
        food_right, food_below = 0, 0

    danger_right = 1 if snakeHead["x"] == 39 else 0
    danger_left = 1 if snakeHead["x"] == 0 else 0
    danger_above = 1 if snakeHead["y"] == 0 else 0
    danger_below = 1 if snakeHead["y"] == 29 else 0
    
    dangerRectRight = copy.deepcopy(listSnakeSquares)
    dangerRectRight[0]["x"] += 1
    dangerRectLeft = copy.deepcopy(listSnakeSquares)
    dangerRectLeft[0]["x"] -= 1
    dangerRectBelow = copy.deepcopy(listSnakeSquares)
    dangerRectBelow[0]["y"] += 1
    dangerRectAbove = copy.deepcopy(listSnakeSquares)
    dangerRectAbove[0]["y"] -= 1

    danger_right = 1 if dangerRectRight[0] in listSnakeSquares[:-1] else danger_right #3[2] [0; 1]
    danger_left = 1 if dangerRectLeft[0] in listSnakeSquares[:-1] else danger_left #4[2] [0; 1]
    danger_above = 1 if dangerRectAbove[0] in listSnakeSquares[:-1] else danger_above #5[2] [0; 1]
    danger_below = 1 if dangerRectBelow[0] in listSnakeSquares[:-1] else danger_below #6[2] [0; 1]
    
    #print(danger_right, danger_left, danger_above, danger_below)
    return (food_right, food_below, danger_right, danger_left,
            danger_above, danger_below) # length_of_snake


DISCRETE_OS_SIZE = [3, 3, 2, 2, 2, 2] #3 * 3 * 2 * 2 * 2 * 2
#q_table = np.random.uniform(low=-1, high=1, size=(DISCRETE_OS_SIZE + [5])) # 5 actions
q_table = np.load(r'C:\Users\Shane Yo\Documents\Coding Projects\Python\pygame\learning\game_n_ai\\q_matrix5.npy')   
LEARNING_RATE = 0.1
DISCOUNT = 0.995
EPISODES = 25000
SHOW_EVERY = 1

epsilon = 0.009
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES//2
epsilon_decay_value = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)

def state_to_action():
    global prev_state, prev_action
    discrete_state = get_states_easier()
    prev_state = discrete_state
    if np.random.random() > epsilon:
        # Get action from Q Table
        action = np.argmax(q_table[discrete_state])
    else:
        # Get random actionS
        action = np.random.randint(0, 5)
    prev_action = action
    return action

# Make the action in between these so states change

def update_q():
    global prev_state, prev_action
    global reward, done
    new_state = get_states_easier()
    add_reward = False
    is_done = False
    if reward:
        add_reward = True
        reward = False
    #if done:
        #is_done = True
        #done = False
    
    # If simulation did not end yet after last step - update Q table
    if not done:    
        # Maximum possible Q value in next step (for new state)
        max_future_q = np.max(q_table[new_state])
        
        # Current Q value (for current state and performed action)
        current_q = q_table[prev_state + (prev_action,)]
        
        # And here's our equation for a new Q value for current state and action
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (add_reward + DISCOUNT * max_future_q)
        
        # Update Q table with new Q value
        q_table[prev_state + (prev_action,)] = new_q
    
    # Simulation ended
    else:
        done = False
        #q_table[discrete_state + (action,)] = reward
        q_table[prev_state + (prev_action,)] = -1
    
    #prev_state = new_state # think this is unnecessary
    
    

###################################QL
def gameLoop():
    global timeSinceLastAction
    global gameOver
    global direction
    if not VISUAL_GAME:
        print_display_surf()
    while True: # main game loop
        if VISUAL_GAME:
            dt = clock.tick() 
            timeSinceLastAction += dt
            #print(timeSinceLastAction)
        
        if not gameOver:
            #print(get_states_easier())
            # dt is measured in milliseconds, therefore 250 ms = 0.25 seconds
            if VISUAL_GAME:
                if timeSinceLastAction > SPEED:
                    try:
                        collide = snakeHead.colliderect(foodRect)
                    except:
                        collide = False
                    action(collide) # move the snake here
                    timeSinceLastAction = 0 # reset it to 0 so you can count again
                    if isDead():
                        gameOver = True
                        continue                       
            else:
                try:
                    collide = (snakeHead == foodRect)
                except:
                    collide = False                  
                    
            
            # If eat food
            if not foodRect:
                changeFood()           
            if VISUAL_GAME:
                DISPLAYSURF.fill(WHITE)
                pygame.draw.rect(DISPLAYSURF, GREEN, foodRect, 0)
                pygame.draw.rect(DISPLAYSURF, BLACK, foodRect, 1)
            else:
                DISPLAYSURF[foodRect["y"]][foodRect["x"]] = "$"
            #pygame.draw.rect(DISPLAYSURF, BLUE, snakeHead, 0)
            #DISPLAYSURF.blit(catImg, (catx, caty))
            #pygame.draw.rect(DISPLAYSURF, BLUE, snakeHead, 0)
            #print(foodRect.x)
            for snakePart in listSnakeSquares:
                if VISUAL_GAME:
                    pygame.draw.rect(DISPLAYSURF, BLUE, snakePart, 0)
                    pygame.draw.rect(DISPLAYSURF, BLACK, snakePart, 1)
                else:
                    try:
                        DISPLAYSURF[snakePart["y"]][snakePart["x"]] = "O"
                    except:
                        break
            
            
        #    if snakeHead.colliderect(foodRect):
        #        foodRect = None        
            if VISUAL_GAME:
                for event in pygame.event.get():
                    #pygame.draw.rect(DISPLAYSURF, BLUE, snakeHead, 0)
                        
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT and direction != 'right':
                            #pass
                            direction = 'left'
                            #snakeHead.move(10,10)
                        elif event.key == pygame.K_RIGHT and direction != 'left':
                            direction = 'right'                
                            #pass
                            #snakeHead.move(10,10)
                        elif event.key == pygame.K_UP and direction != 'down':
                            direction = 'up'   
                       #     snakeHead.move(10,10)
                        elif event.key == pygame.K_DOWN and direction != 'up':
                            direction = 'down'
                      #      snakeHead.move(10,10)
                            
                    elif event.type == QUIT:
                        pygame.quit()
                        sys.exit()
            else:
                move = input("Enter your next move (w,s,a,d,n): ")
                move = move.lower()
                if move == "a" and direction != 'right':
                    #pass
                    direction = 'left'
                    #snakeHead.move(10,10)
                elif move == "d" and direction != 'left':
                    direction = 'right'                
                    #pass
                    #snakeHead.move(10,10)
                elif move == "w" and direction != 'down':
                    direction = 'up'   
               #     snakeHead.move(10,10)
                elif move == "s" and direction != 'up':
                    direction = 'down'
                action(collide) # move the snake here
                timeSinceLastAction = 0 # reset it to 0 so you can count again
                if isDead():
                    gameOver = True
                    continue               
        else:
            if VISUAL_GAME:
                DISPLAYSURF.fill(RED)
            FinalText('Game Over! Your score was {}'.format(score))
            if VISUAL_GAME:
                for event in pygame.event.get():
                    #print('========IN HERE==============')
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            reset()
                            continue
                        
                    elif event.type == QUIT:
                        pygame.quit()
                        sys.exit()
            else:
                play_again = input("Do you want to play again (Y/N)?: ")
                play_again = play_again.lower()
                if play_again == "y":
                    reset()
                    continue
                else:
                    sys.exit()
                    
            
        #print(get_states_easier())
        if VISUAL_GAME:
            pygame.display.update()
        else:
            print_display_surf()
        #fpsClock.tick(FPS)

MAX_SCORE = 0

################################ MORE QL #######################
def simulation_gameLoop():
    global gameOver
    global direction
    global epsilon
    global MAX_SCORE
    
    move_map = {0:"w",1:"s",2:"a", 3:"d", 4:"n"}
    #print_display_surf()
    for episode in range(EPISODES):
        print(f"EPISODE: {episode + 1} of {EPISODES}")
        render = (episode % SHOW_EVERY == 0) #or (episode > 12500)
        #render = True if episode > 7500 else render
        #how_long_collided = 0
        counter = 0
        prev_epsilon = copy.deepcopy(epsilon)
        while True: # main simulation game loop
            reward = 0
            discrete_state = get_states_easier()
            if not gameOver:
                done = False
                try:
                    collide = (snakeHead == foodRect)
                except:
                    collide = False                  
                
                # If eat food
                if not foodRect:
                    changeFood()           

                if DISPLAYSURF:
                    DISPLAYSURF[foodRect["y"]][foodRect["x"]] = "$"

                for snakePart in listSnakeSquares:
                    try:
                        if DISPLAYSURF:
                            DISPLAYSURF[snakePart["y"]][snakePart["x"]] = "O"
                    except:
                        break
                
                current_action = state_to_action()
                move = move_map[current_action]
                if not counter:
                    move = "s"
                counter += 1
                #if counter > 10000:
                #    render = True
                if move == "a" and direction != 'right':
                    #pass
                    direction = 'left'
                    #snakeHead.move(10,10)
                elif move == "d" and direction != 'left':
                    direction = 'right'                
                    #pass
                    #snakeHead.move(10,10)
                elif move == "w" and direction != 'down':
                    direction = 'up'   
               #     snakeHead.move(10,10)
                elif move == "s" and direction != 'up':
                    direction = 'down'
                prev_listSnakeSquares = copy.deepcopy(listSnakeSquares)
                prev_distance = ((foodRect['x'] - snakeHead['x'])**2 + (foodRect['y'] - snakeHead['y'])**2)**0.5 if foodRect else None
                action(collide) # move the snake here
                new_distance = ((foodRect['x'] - snakeHead['x'])**2 + (foodRect['y'] - snakeHead['y'])**2)**0.5 if foodRect else None
                new_discrete_state = get_states_easier()
                if collide:
                    counter = 1
                    #reward += score**3
                    #reward = 5**(score//10)
                    reward = 1
                    prev_distance = None
    
                else:
                    if foodRect:
                        #reward -= (round(((foodRect['x'] - snakeHead['x'])**2 + (foodRect['y'] - snakeHead['y'])**2)**0.5)/(WIDTH + HEIGHT)) 
                        #reward = -counter
                        #reward -= counter / 20000
                        if new_distance is not None and prev_distance is not None:
                            if new_distance < prev_distance:
                                reward = 0.1
                            else:
                                reward = -0.1

                if sorted(listSnakeSquares, key=lambda d: (d['x'],d['y'])) == sorted(prev_listSnakeSquares, key=lambda d: (d['x'],d['y'])):
                    #reward -= (score**2)
                    reward = -1

                if counter > 10000:
                    #reward -= ((counter/50)**2)
                    reward = -1
                    epsilon += epsilon_decay_value
                    

        
                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(q_table[new_discrete_state])
    
                # Current Q value (for current state and performed action)
                current_q = q_table[discrete_state + (current_action,)]
    
                # And here's our equation for a new Q value for current state and action
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    
                # Update Q table with new Q value
                #q_table[discrete_state + (current_action,)] = new_q

                if render:
                    time.sleep(0.05)

                if isDead():
                    gameOver = True
                    if score > MAX_SCORE:
                        MAX_SCORE = score
                        #with open(r'C:\Users\Shane Yo\Documents\Coding Projects\Python\pygame\learning\q_matrix5.npy', 'wb') as f:
                            #np.save(f, q_table)
                            #print("MAX SCORE")
                            #print(MAX_SCORE)
                    #reward -= ((WIDTH * HEIGHT) - len(listSnakeSquares)) /(WIDTH * HEIGHT)
                    #reward = -5
                    #q_table[discrete_state + (current_action,)] -= 5
                    continue               
            else:
                #FinalText('Game Over! Your score was {}'.format(score))
                print(f"Game Over! Your score was {score}")
                reset(render)
                break
                    
            if render:
                print_display_surf(episode)
        print(f"epsilon: {epsilon}")
        print(counter)
        print(MAX_SCORE)
        epsilon = copy.deepcopy(prev_epsilon)
        if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
            epsilon -= epsilon_decay_value

def simulation():
    for episode in range(EPISODES):
        discrete_state = print(get_states_easier())
        done = False
    
        while not done:
    
            action = state_to_action()
    
    
            new_state, reward, done, _ = env.step(action)
    
            new_discrete_state = get_discrete_state(new_state)
    
            if episode % SHOW_EVERY == 0:
                env.render()
            #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    
            # If simulation did not end yet after last step - update Q table
            if not done:
    
                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(q_table[new_discrete_state])
    
                # Current Q value (for current state and performed action)
                current_q = q_table[discrete_state + (action,)]
    
                # And here's our equation for a new Q value for current state and action
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    
                # Update Q table with new Q value
                q_table[discrete_state + (action,)] = new_q
    
    
            # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
            else:
                #q_table[discrete_state + (action,)] = reward
                q_table[discrete_state + (action,)] = -1
    
            discrete_state = new_discrete_state
    
        # Decaying is being done every episode if episode number is within decaying range
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_values
            
##def random(x, y,prev_x=1,prev_y=1,new_iter=0,test={}):
##    if prev_x == x and prev_y == y:
##        if (prev_x, prev_y) not in test:
##            test[(prev_x,prev_y)] = new_iter
##        else:
##            if new_iter < test[(prev_x,prev_y)]:
##                test[(prev_x,prev_y)] = new_iter
##    if prev_x > x or prev_y > y:
##        return
##     random(x, y,prev_x=prev_x + prev_y,prev_y=prev_y,new_iter=new_iter+1,test=test)
##     random(x, y,prev_x=prev_x,prev_y=prev_x+prev_y,new_iter=new_iter+1,test=test)
    
        

###############################################################

if __name__ == '__main__':
    if AI:
        simulation_gameLoop()
    else:
        gameLoop()
    
    
    
