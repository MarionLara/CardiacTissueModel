import numpy as np
import tqdm
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
import scipy as sp
import scipy.io as sio
from matplotlib.animation import FuncAnimation
import csv

number_of_runs = 5

for main_index in range(1,number_of_runs+1):
  run_number = str(main_index)

  #brownianMat = [[0.48,0.09],[0.09,0.04]] #approximately those found by mathematica
  brownianMat = [[0.5, 0.125],[0.125,0]]
  brownianLen = 1
  brownianLimit = 3
  probLimit = 0.000001
  env_length = 0
  env_height = 0
  inside_baseline = 3
  outside_baseline = 10
  jump = 1
  transmemberane_breakpoint = 0 #outside - inside < this
  D = 5.23e-6  # cm^2/ms, value taken from Suenson et al. (1974)
  time_scale = 400000 #seems to have a sweet spot ~300000 that minimizes noise while converging to equilibrium; minimum of 1/2D to make sense, ideally higher
  potential_scale = 3 #for a desired minimum guaranteed movement potential L, set this to 2*D*L*time_scale
  rng = np.random.default_rng(12345)

  env = [[]]
  state = [[]]
  cells = [[]]
  fullBrownianMat = [[]]
  max_length = np.sqrt(time_scale * D)
  membraneUpdate = False

  def createEnvironment(cell_length, cell_height, cell_gap_hor, cell_gap_vert, cell_number):
    global env
    global env_length
    global env_height
    global jump
    global cells
    env_length = (cell_length + cell_gap_hor) * cell_number + cell_gap_hor
    env_height = cell_height + 2 * cell_gap_vert
    env = [[[] for _ in range(env_length)] for __ in range(env_height)]
    jump = cell_gap_hor + 1
    for j in range(len(env)):
      for i in range(len(env[j])):
        if i == 0:
          env[j][i].append(('perm', 'left'))
        if i == env_length - 1:
          env[j][i].append(('perm', 'right'))
        if j == 0:
          env[j][i].append(('perm', 'down'))
        if j == env_height - 1:
          env[j][i].append(('perm', 'up'))
        if i % (cell_length + cell_gap_hor) == cell_gap_hor - 1 and j >= cell_gap_vert and j < env_height - cell_gap_vert and ('closed','right') not in env[j][i] and ('perm','right') not in env[j][i]:
          env[j][i].append(('closed', 'right'))
        if i % (cell_length + cell_gap_hor) == cell_length + cell_gap_hor - 1 and j >= cell_gap_vert and j < env_height - cell_gap_vert and ('closed','right') not in env[j][i] and ('perm','right') not in env[j][i]:
          env[j][i].append(('perm', 'right'))
        if i % (cell_length + cell_gap_hor) == 0 and j >= cell_gap_vert and j < env_height - cell_gap_vert and ('closed','left') not in env[j][i] and ('perm','left') not in env[j][i]:
          env[j][i].append(('closed', 'left'))
        if i % (cell_length + cell_gap_hor) == cell_gap_hor and j >= cell_gap_vert and j < env_height - cell_gap_vert and ('closed','left') not in env[j][i] and ('perm','left') not in env[j][i]:
          env[j][i].append(('perm', 'left'))
        if j % (cell_height + cell_gap_vert) == cell_gap_vert - 1 and i % (cell_length + cell_gap_hor) >= cell_gap_hor and ('closed','up') not in env[j][i] and ('perm','up') not in env[j][i]:
          env[j][i].append(('closed', 'up'))
        if j % (cell_height + cell_gap_vert) == cell_height + cell_gap_vert - 1 and i % (cell_length + cell_gap_hor) >= cell_gap_hor and ('closed','up') not in env[j][i] and ('perm','up') not in env[j][i]:
          env[j][i].append(('perm','up'))
        if j % (cell_height + cell_gap_vert) == 0 and i % (cell_length + cell_gap_hor) >= cell_gap_hor and ('closed','down') not in env[j][i] and ('perm','down') not in env[j][i]:
          env[j][i].append(('closed', 'down'))
        if j % (cell_height + cell_gap_vert) == cell_gap_vert and i % (cell_length + cell_gap_hor) >= cell_gap_hor and ('closed','down') not in env[j][i] and ('perm','down') not in env[j][i]:
          env[j][i].append(('perm', 'down'))
        if i % (cell_length + cell_gap_hor) == cell_length + cell_gap_hor - 1 and j >= cell_gap_vert and j <= env_height - cell_gap_vert and i < env_length - cell_gap_hor - 1:
          env[j][i].append(('gap', 'right'))
        if i % (cell_length + cell_gap_hor) == cell_gap_hor and j >= cell_gap_vert and j <= env_height - cell_gap_vert and i > cell_gap_hor:
          env[j][i].append(('gap', 'left'))

    cells = [[0 for _ in range(env_length)] for __ in range(env_height)]
    for j in range(len(env)):
      for i in range(len(env[j])):
        if i % (cell_length + cell_gap_hor) >= cell_gap_hor and i % (cell_length + cell_gap_hor) <= cell_length + cell_gap_hor - 1 and j >= cell_gap_vert and j < env_height - cell_gap_vert:
          cells[j][i] = 1


  def createInitialization():
    global state
    state = np.zeros((env_height,env_length), dtype = int)

    # for j in range(env_height):
    #   for i in range(env_length):
    #     mean = 0
    #     if cells[j][i] == 0:
    #       mean = outside_baseline
    #     elif cells[j][i] == 1:
    #       mean = inside_baseline
    #     state[j][i] = max(mean + rng.integers(-2,2), 0)

    #state = np.append(state[:, 0:1], np.append(30 * np.ones((env_height,1),dtype = int), state[:, 2:], axis = 1), axis=1)

    #state = np.append(10 * np.ones((env_height,1),dtype = int), state[:, 1:], axis=1)

    #state = np.array([[rng.integers(5) for _ in range(env_length)] for __ in range(env_height)])

    for j in range(env_height):
      for i in range(env_length):
        if ('closed','right') in env[j][i] or ('closed','up') in env[j][i] or ('closed','left') in env[j][i] or ('closed','down') in env[j][i]:
          state[j][i] = max(80 + np.random.normal(0, 4), 0)

    #set up intialization



  def pokeHole():
    global env
    middle = int(env_height/2)
    for i in env[middle]:
      if ('closed', 'right') in i:
        i.remove(('closed', 'right'))
        i.append(('open', 'right'))
        return

  cell_length, cell_height, extra_length, extra_height, cell_number = 24, 4, 1, 1, 3


  createEnvironment(cell_length, cell_height, extra_length, extra_height, cell_number)
  createInitialization()
  pokeHole()
  #print(state)
  #print(env_length, env_height)
  #print(*env, sep="\n")

  ###detects boundaries when calculating gradient

  def boundaryFixerRight(x, state):
    #gives the potential in the right neighborhood, possibly fixing it for the special cases (boundaries, gap junctions)
    if ('closed','right') not in env[x[0]][x[1]] and ('perm', 'right') not in env[x[0]][x[1]]:
      return state[x[0],x[1]+1]
    elif not ('gap', 'right') in env[x[0]][x[1]]:
      return state[x[0],x[1]]
    elif ('gap', 'right') in env[x[0]][x[1]]:
      return state[x[0],x[1]+jump]


  def boundaryFixerLeft(x, state):
    #gives the potential in the left neighborhood, possibly fixing it for the special cases (boundaries, gap junctions)
    if ('closed','left') not in env[x[0]][x[1]] and ('perm', 'left') not in env[x[0]][x[1]]:
      return state[x[0],x[1]-1]
    elif not ('gap', 'left') in env[x[0]][x[1]]:
      return state[x[0],x[1]]
    elif ('gap', 'left') in env[x[0]][x[1]]:
      return state[x[0],x[1]-jump]


  def boundaryFixerUp(x, state):
    #gives the potential in the up neighborhood, possibly fixing it for the special cases (boundaries, gap junctions)
    if ('closed','up') not in env[x[0]][x[1]] and ('perm', 'up') not in env[x[0]][x[1]]:
      return state[x[0]+1,x[1]]
    elif not ('gap', 'up') in env[x[0]][x[1]]:
      return state[x[0],x[1]]
    elif ('gap', 'up') in env[x[0]][x[1]]:
      return state[x[0]+jump,x[1]]


  def boundaryFixerDown(x, state):
    #gives the potential in the down neighborhood, possibly fixing it for the special cases (boundaries, gap junctions)
    if ('closed','down') not in env[x[0]][x[1]] and ('perm', 'down') not in env[x[0]][x[1]]:
      return state[x[0]-1,x[1]]
    elif not ('gap', 'down') in env[x[0]][x[1]]:
      return state[x[0],x[1]]
    elif ('gap', 'down') in env[x[0]][x[1]]:
      return state[x[0]-jump,x[1]]

  ###detects boundaries when ions move

  def boundaryShiftRight(vector, pos): #vector should have positive x value (actually above -0.5)
    if vector == [0,0]: #vector in y-x, pos in y-x
      return [0,0]
    elif vector[1] <= 0.5 and vector[1] >= -0.5:
      return boundaryShiftVert([vector[0],0], pos)
    elif vector[1] < abs(vector[0]):
      return boundaryShiftVert(vector, pos) #will zig-zag if exactly 45 degree slope
    elif (('closed','right') in env[pos[0]][pos[1]] or ('perm', 'right') in env[pos[0]][pos[1]]) and not ('gap', 'right') in env[pos[0]][pos[1]]: #moves right until hit 45 degree from target, then zig-zag
      return boundaryShiftVert([vector[0],0], pos)
    elif (('closed','right') in env[pos[0]][pos[1]] or ('perm', 'right') in env[pos[0]][pos[1]]) and ('gap', 'right') in env[pos[0]][pos[1]]:
      temp = boundaryShiftRight([vector[0], vector[1]-1],[pos[0], pos[1]+jump])
      return [temp[0], temp[1]+jump]
    else:
      temp = boundaryShiftRight([vector[0], vector[1]-1], [pos[0],pos[1]+1])
      return [temp[0], temp[1]+1]

  def boundaryShiftLeft(vector,pos): #vector should have negative x value
    if vector == [0,0]:
      return [0,0]
    elif vector[1] <= 0.5 and vector[1] >= -0.5:
      return boundaryShiftVert([vector[0],0], pos)
    elif -vector[1] < abs(vector[0]):
      return boundaryShiftVert(vector, pos)
    elif (('closed','left') in env[pos[0]][pos[1]] or ('perm', 'left') in env[pos[0]][pos[1]]) and not ('gap', 'left') in env[pos[0]][pos[1]]:
      return boundaryShiftVert([vector[0],0], pos)
    elif (('closed','left') in env[pos[0]][pos[1]] or ('perm', 'left') in env[pos[0]][pos[1]]) and ('gap', 'left') in env[pos[0]][pos[1]]:
      temp = boundaryShiftLeft([vector[0], vector[1]+1],[pos[0], pos[1]-jump])
      return [temp[0], temp[1] - jump]
    else:
      temp = boundaryShiftLeft([vector[0], vector[1]+1], [pos[0],pos[1]-1])
      return [temp[0], temp[1]-1]

  def boundaryShiftUp(vector, pos): #vector should have positive y value
    if vector == [0,0]:
      return [0,0]
    elif vector[0] <= 0.5 and vector[0] >= -0.5:
      return boundaryShiftHor([0,vector[1]], pos)
    elif abs(vector[1]) > vector[0]:
      return boundaryShiftHor(vector, pos)
    elif (('closed','up') in env[pos[0]][pos[1]] or ('perm', 'up') in env[pos[0]][pos[1]]) and not ('gap', 'up') in env[pos[0]][pos[1]]:
      return boundaryShiftHor([0,vector[1]], pos)
    elif (('closed','up') in env[pos[0]][pos[1]] or ('perm', 'up') in env[pos[0]][pos[1]]) and ('gap', 'up') in env[pos[0]][pos[1]]:
      temp = boundaryShiftUp([vector[0] - 1, vector[1]],[pos[0]+jump, pos[1]])
      return [temp[0] + jump, temp[1]]
    else:
      temp = boundaryShiftUp([vector[0] - 1, vector[1]], [pos[0] + 1 ,pos[1]])
      return [temp[0]+1, temp[1]]

  def boundaryShiftDown(vector, pos): #vector should have negative y value
    if vector == [0,0]:
      return [0,0]
    elif vector[0] <= 0.5 and vector[0] >= -0.5:
      return boundaryShiftHor([0,vector[1]], pos)
    elif abs(vector[1]) > -vector[0]:
      return boundaryShiftHor(vector, pos)
    elif (('closed','down') in env[pos[0]][pos[1]] or ('perm', 'down') in env[pos[0]][pos[1]]) and not ('gap', 'down') in env[pos[0]][pos[1]]:
      return boundaryShiftHor([0,vector[1]], pos)
    elif (('closed','down') in env[pos[0]][pos[1]] or ('perm', 'down') in env[pos[0]][pos[1]]) and ('gap', 'down') in env[pos[0]][pos[1]]:
      temp = boundaryShiftUp([vector[0] + 1, vector[1]],[pos[0] - jump, pos[1]])
      return [temp[0] - jump, temp[1]]
    else:
      temp = boundaryShiftUp([vector[0] + 1, vector[1]], [pos[0] - 1,pos[1]])
      return [temp[0] - 1, temp[1]]

  def boundaryShiftHor(vector, pos):
    if vector[1] >= 0:
      return boundaryShiftRight(vector,pos)
    else:
      return boundaryShiftLeft(vector,pos)

  def boundaryShiftVert(vector,pos):
    if vector[0] >= 0:
      return boundaryShiftUp(vector,pos)
    else:
      return boundaryShiftDown(vector,pos)

  ### updates membranes

  def membraneUpdateRight(pos, state, membrane):
    if ('closed', 'right') not in membrane:
      return membrane
    elif state[pos[0]][pos[1] + 1] - state[pos[0]][pos[1]] > transmemberane_breakpoint:
      membrane.remove(('closed','right'))
      membrane.append(('open','right'))
      global membraneUpdate
      membraneUpdate = True
      return membrane
    else:
      return membrane

  def membraneUpdateLeft(pos, state, membrane):
    if ('closed', 'left') not in membrane:
      return membrane
    elif state[pos[0]][pos[1] - 1] - state[pos[0]][pos[1]] > transmemberane_breakpoint:
      membrane.remove(('closed','left'))
      membrane.append(('open', 'left'))
      global membraneUpdate
      membraneUpdate = True
      return membrane

    else:
      return membrane

  def membraneUpdateUp(pos, state, membrane):
    if ('closed', 'up') not in membrane:
      return membrane
    elif state[pos[0] + 1][pos[1]] - state[pos[0]][pos[1]] > transmemberane_breakpoint:
      membrane.remove(('closed','up'))
      membrane.append(('open', 'up'))
      global membraneUpdate
      membraneUpdate = True
      return membrane
    else:
      return membrane

  def membraneUpdateDown(pos, state, membrane):
    if ('closed', 'down') not in membrane:
      return membrane
    elif state[pos[0] - 1][pos[1]] - state[pos[0]][pos[1]] > transmemberane_breakpoint:
      membrane.remove(('closed','down'))
      membrane.append(('open', 'down'))
      global membraneUpdate
      membraneUpdate = True
      return membrane
    else:
      return membrane

  def membraneUpdateAll(pos, state):
    return membraneUpdateRight(pos, state, membraneUpdateLeft(pos, state, membraneUpdateUp(pos, state, membraneUpdateDown(pos, state, env[pos[0]][pos[1]]))))

  def fullMembraneUpdate(state):
    global env
    env1 = [[membraneUpdateAll([j,i], state) for i in range(env_length)] for j in range(env_height)]
    env = env1

  ###

  def delta(size):
    #result = min(np.sqrt(time_scale * size * D), max_length)
    result = max_length
    return result

  def gradVector(down, up, left, right):
    vert = down - up
    hor = left - right
    norm = np.sqrt(vert**2 + hor**2)
    p = norm / potential_scale
    if norm != 0:
      return [delta(p) * vert / norm, delta(p) * hor / norm]
    else:
      return [0,0]

  ###detects boundaries for brownian motion

  def brownianFixerRight(x):
    #goes from right nbd to original nbd, so it moves left
    try:
      if ('closed','left') not in env[x[0]][x[1]] and ('perm', 'left') not in env[x[0]][x[1]]:
        return brownianMat[0][1]
      else:
        return 0
    except:
      return 0

  def brownianFixerLeft(x):
    try:
      if ('closed','right') not in env[x[0]][x[1]] and ('perm', 'right') not in env[x[0]][x[1]]:
        return brownianMat[0][1]
      else:
        return 0
    except:
      return 0

  def brownianFixerUp(x):
    try:
      if ('closed','down') not in env[x[0]][x[1]] and ('perm', 'down') not in env[x[0]][x[1]]:
        return brownianMat[1][0]
      else:
        return 0
    except:
      return 0

  def brownianFixerDown(x):
    try:
      if ('closed','up') not in env[x[0]][x[1]] and ('perm', 'up') not in env[x[0]][x[1]]:
        return brownianMat[1][0]
      else:
        return 0
    except:
      return 0

  def brownianDownRightCorner(x):
    #moves left first, then up
    try:
      if ('closed', 'left') not in env[x[0] + 1][x[1]] and ('closed', 'left') not in env[x[0] + 1][x[1]]:
        if ('closed', 'up') not in env[x[0]][x[1]] and ('perm', 'up') not in env[x[0]][x[1]]:
          return brownianMat[1][1]
      return 0
    except:
      return 0

  def brownianDownLeftCorner(x):
    try:
      if ('closed', 'right') not in env[x[0] + 1][x[1]] and ('closed', 'right') not in env[x[0] + 1][x[1]]:
        if ('closed', 'up') not in env[x[0]][x[1]] and ('perm', 'up') not in env[x[0]][x[1]]:
          return brownianMat[1][1]
      return 0
    except:
      return 0

  def brownianUpRightCorner(x):
    try:
      if ('closed', 'left') not in env[x[0] - 1][x[1]] and ('closed', 'left') not in env[x[0] - 1][x[1]]:
        if ('closed', 'down') not in env[x[0]][x[1]] and ('perm', 'down') not in env[x[0]][x[1]]:
          return brownianMat[1][1]
      return 0
    except:
      return 0

  def brownianUpLeftCorner(x):
    try:
      if ('closed', 'right') not in env[x[0] - 1][x[1]] and ('closed', 'right') not in env[x[0] - 1][x[1]]:
        if ('closed', 'down') not in env[x[0]][x[1]] and ('perm', 'down') not in env[x[0]][x[1]]:
          return brownianMat[1][1]
      return 0
    except:
      return 0

  def brownianFixerCurrent(x):
    total = 0
    try:
      if ('closed', 'right') in env[x[0]][x[1]] or ('perm', 'right') in env[x[0]][x[1]]:
        total += brownianMat[0][1]
      if ('closed', 'left') in env[x[0]][x[1]] or ('perm', 'left') in env[x[0]][x[1]]:
        total += brownianMat[0][1]
      if ('closed', 'up') in env[x[0]][x[1]] or ('perm', 'up') in env[x[0]][x[1]]:
        total += brownianMat[1][0]
      if ('closed', 'down') in env[x[0]][x[1]] or ('perm', 'down') in env[x[0]][x[1]]:
        total += brownianMat[1][0]
      total += brownianMat[0][0]
      return total
    except:
      return total


  def brownianGenericFixer(pos, x):
    #brownian motion going from pos in direction x
    if x[1] == 0 and x[0] == 0:
      return brownianFixerCurrent(pos)
    elif x[1] > 0 and x[0] == 0:
      return brownianFixerRight(pos)
    elif x[1] < 0 and x[0] == 0:
      return brownianFixerLeft(pos)
    elif x[1] == 0 and x[0] > 0:
      return brownianFixerUp(pos)
    elif x[1] == 0 and x[0] < 0:
      return brownianFixerDown(pos)
    elif x[1] > 0 and x[0] > 0:
      return brownianUpRightCorner(pos)
    elif x[1] < 0 and x[0] > 0:
      return brownianUpLeftCorner(pos)
    elif x[1] > 0 and x[0] < 0:
      return brownianDownRightCorner(pos)
    elif x[1] < 0 and x[0] < 0:
      return brownianDownLeftCorner(pos)
    else:
      return 0

  def BrownianHelper(pos_start, pos_target):
    diff = [-pos_start[0] + pos_target[0], -pos_start[1] + pos_target[1]]
    if abs(diff[0]) > 1 or abs(diff[1]) > 1:
      return 0 #no brownian motion across gap junctions
    else:
      return brownianGenericFixer(pos_target, diff)

  def updateBrownianMotion():
    #only updates if env actually changed
    #turns state matrix into a vector, computes a matrix that will multiply to induce Brownian motion
    global fullBrownianMat
    mat = [[BrownianHelper([ int(x / env_length), x % env_length], [int(y / env_length), y % env_length]) for x in range(env_height * env_length)] for y in range(env_height * env_length)]
    #flatten so x%env_height is the horizontal position, x/env_height is the vertical position

    fullBrownianMat = np.array(mat)


  def brownianMotion(state):
    #does the matrix multiplication
    temp_state = [state[int(i/env_length)][i % env_length] for i in range(env_height * env_length)]
    temp2_state = fullBrownianMat.T @ np.array(temp_state)
    new_state = [[temp2_state[j*env_length + i] for i in range(env_length)] for j in range(env_height)]
    return np.array(new_state)

  updateBrownianMotion()
  #print(brownianMotion(state))

  def PS(state):
    #compute movement
    movement_vector = [[gradVector(boundaryFixerDown([y,x] ,state) , boundaryFixerUp([y,x], state), boundaryFixerLeft([y,x] ,state) , boundaryFixerRight([y,x], state)) for x in range(env_length)] for y in range(env_height)]
    #compute any shifting from boundaries
    movement_vector_fixed = [[boundaryShiftHor(movement_vector[j][i], [j,i]) for i in range(env_length)] for j in range(env_height)]
    new_state = np.zeros_like(state)
    for j in range(env_height):
      for i in range(env_length):
        new_state[j + movement_vector_fixed[j][i][0], i + movement_vector_fixed[j][i][1]] += state[j,i]
    #return mean resulting from BM
    global membraneUpdate
    if membraneUpdate:
      updateBrownianMotion()
      membraneUpdate = False
    newer_state = np.array(brownianMotion(new_state))
    return(newer_state)

  cell_differences = []
  cell_differences2 = []
  column_values = []
  cell1_sum = []
  cell2_sum = []
  cell3_sum = []
  rightmost_open_membrane = []
  leftmost_closed_membrane = []
  temp = 0
  temp2 = 0

  # all_cell_sums = [[] for i in range(10)]

  cell1_sum.append(np.sum(state[extra_height:cell_height + extra_height, extra_length:cell_length + extra_length]))
  cell2_sum.append(np.sum(state[extra_height:cell_height + extra_height, 2 * extra_length + cell_length:2 * (cell_length + extra_length)]))
  cell3_sum.append(np.sum(state[extra_height:cell_height + extra_height, 3 * extra_length + 2 * cell_length:3 * (cell_length + extra_length)]))

  # for i in range(1,11):
  #   all_cell_sums[i-1].append(np.sum(state[extra_height:cell_height + extra_height,
  #                                  i * extra_length + (i - 1) * cell_length:i * (cell_length + extra_length)]))

  for _ in tqdm.tqdm(range(500)):
    state = PS(state)
    fullMembraneUpdate(state)

    #print(state)
    #print(np.sum(state))

    cell1_sum.append(np.sum(state[extra_height:cell_height+extra_height, extra_length:cell_length+extra_length]))
    cell2_sum.append(np.sum(state[extra_height:cell_height+extra_height,2*extra_length+cell_length:2*(cell_length+extra_length)]))
    cell3_sum.append(np.sum(state[extra_height:cell_height+extra_height, 3*extra_length+2*cell_length:3*(cell_length+extra_length)]))

    # for i in range(1, 11):
    #   all_cell_sums[i-1].append(np.sum(state[extra_height:cell_height+extra_height, i*extra_length+(i-1)*cell_length:i*(cell_length+extra_length)]))

    #cell_differences.append(np.sum(state[:,0]) - np.sum(state[:,env_length - 1]))
    #cell_differences2.append(np.sum(state[:, 0:1]) - np.sum(state[:, env_length - 2 : env_length-1]))
    #cell_differences.append(np.sum(state[1:env_height-2,9]))
    #cell_differences2.append(np.sum(state[1:env_height-2,2]))

    #column_values.append([np.sum(state[:,i]) for i in range(env_length)])
    if temp != env_length - extra_length:
      for i in range(env_length - 1, -1, -1):
        for j in range(env_height):
          if ('open', 'right') in env[j][i] or ('open', 'left') in env[j][i] or ('open', 'up') in env[j][i] or ('open', 'down') in env[j][i]:
            temp = i
            break
        else:
          continue
        break
      rightmost_open_membrane.append(temp)
    if temp2 < env_length - extra_length:
      for i in range(env_length):
        for j in range(env_height):
          if ('closed', 'right') in env[j][i] or ('closed', 'left') in env[j][i] or ('closed', 'up') in env[j][i] or ('closed', 'down') in env[j][i] or i >= env_length - extra_length:
            temp2 = i
            break
        else:
          continue
        break
      leftmost_closed_membrane.append(temp2)


  def averageList(list, k):
    return [np.sum([list[i] for i in range(j, min(j + k, len(list) - 1))]) / k for j in range(len(list))]

  #time_steps = range(len(cell_differences_averaged))
  #plt.plot(time_steps, cell_differences_averaged, label='end of cell')
  #plt.plot(time_steps, cell_differences_averaged2, label='start of cell')
  #plt.plot(time_steps, cell_differences)
  #plt.plot(time_steps, cell_differences2)
  #plt.legend()
  #plt.show()


  # celltotal_sum = [cell1_sum[i] + cell2_sum[i] + cell3_sum[i] for i in range(len(cell1_sum))]
  # time_steps = range(len(cell1_sum))
  # plt.plot(time_steps, averageList(cell1_sum,4), label='cell 1')
  # plt.plot(time_steps, averageList(cell2_sum,4), label='cell 2')
  # plt.plot(time_steps, averageList(cell3_sum,4), label='cell 3')
  # plt.plot(time_steps, celltotal_sum, label = 'total')
  # plt.legend()
  # plt.show()

  #time_steps = range(len(rightmost_open_membrane))
  #plt.plot(time_steps, rightmost_open_membrane, label='wavefront')
  #plt.legend()
  #plt.show()

  # length = [i for i in range(env_length)]
  # fig, ax = plt.subplots()
  # def animate(i):
  #   x = length
  #   y = column_values[5*i]
  #   ax.clear()
  #   ax.plot(x, y)
  #   ax.set_ylim([0,50])
  #ani = FuncAnimation(fig, animate, frames=int(len(column_values)/5), interval=100, repeat=False)
  #plt.show()

  #space_steps = [i for i in range(env_length)]
  #plt.plot(space_steps, column_values[len(column_values)-1])
  #plt.show()

  # celltotal_sum = [sum([j[i] for j in all_cell_sums]) for i in range(len(cell1_sum))]
  # time_steps = range(len(celltotal_sum))
  # for i in range(len(all_cell_sums)):
  #   plt.plot(time_steps, averageList(all_cell_sums[i], 4), label = 'Cell ' + str(i+1))
  # plt.plot(time_steps, celltotal_sum, label = 'total')
  # plt.legend()
  # plt.show()

  # print(state)


  time_steps = range(len(rightmost_open_membrane))
  coefficientsO = np.polyfit(time_steps, rightmost_open_membrane, 1)
  p = np.poly1d(np.polyfit(time_steps, rightmost_open_membrane, 1))
  # x_line = np.linspace(np.amin(time_steps), np.amax(time_steps), 200)
  # plt.scatter(time_steps, rightmost_open_membrane)
  # plt.plot(x_line, p(x_line), label = p)
  # print(p)
  # plt.legend()
  # plt.show()

  time_steps2 = range(len(leftmost_closed_membrane))
  coefficientsC = np.polyfit(time_steps2, leftmost_closed_membrane, 1)
  p2 = np.poly1d(np.polyfit(time_steps2, leftmost_closed_membrane, 1))

  # x_line = np.linspace(np.amin(time_steps2), np.amax(time_steps2), 200)
  # plt.scatter(time_steps, rightmost_open_membrane)
  # plt.scatter(time_steps2, leftmost_closed_membrane)
  # plt.plot(x_line, p(x_line), label = p)
  # plt.plot(x_line, p2(x_line), label = p2)
  # plt.legend()
  # plt.show()


  with open('model_data' + run_number + '.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    # for i in range(10):
    #   writer.writerow(['Cell ' + str(i+1) + ' sums'] + all_cell_sums[i])

    writer.writerow(['Cell 1 sums'] + cell1_sum)
    writer.writerow(['Cell 2 sums'] + cell2_sum)
    writer.writerow(['Cell 3 sums'] + cell3_sum)

    writer.writerow(['Rightmost open membrane'] + rightmost_open_membrane)
    writer.writerow(['Leftmost closed membrane'] + leftmost_closed_membrane)
    writer.writerow(['Regression upper'] + coefficientsO.tolist())
    writer.writerow(['Regression lower'] + coefficientsC.tolist())
    writer.writerow(['Time scale'] + [time_scale])
    writer.writerow([])

