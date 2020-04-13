import numpy as np
import time
import pdb
import random
import math

ROWS = 8
COLS = 8
HEADINGS = 4
#directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
directions = {}
directions[0] = [0,1]
directions[1] = [1,0]
directions[2] = [0, -1]
directions[3] = [-1, 0]

no_reading_obs_matrix = np.ones((ROWS*COLS*HEADINGS))*0.325

def main():
	Tr = transition_matrix()
	All_Observations = build_observation_matrix()
	t = 100 #Number of times we'll follow the robot
	ran = False #Change to True to try random
	sensor_only = False #Change to True and ran to False to test sensor only

	for i in range(3):
		if i == 0:
			print("USING THE FORWARD ALGORITHM\n")
			
		elif i == 1:
			sensor_only = True
			print("USING SENSOR READ ONLY\n")
		else:
			sensor_only = False
			ran = True
			print("USING ONLY RANDOM GUESSES\n")
			



	
		tot_man_dist = [] 
		correct_guesses = 0
		for i in range(100):
			rob = Robot()
			pred_state = random.randint(0,ROWS * COLS * HEADINGS - 1)
			sensor_guess = All_Observations[random.randint(0,ROWS * COLS)]
			#print("Iteration:", i, " ...")
			f = np.zeros((ROWS*COLS*HEADINGS, t + 1))
			f[:, 0] = 1/(ROWS*COLS*HEADINGS)
			pred_board = np.zeros((ROWS, ROWS))
			man_dist = []
			for i in range(1,t - 1, 1):
				rob.move_robot()

				
				idx = rob.obtain_sensor_reading() #Actual robot position
				col = idx[0][0] * ROWS + idx[0][1] #Column for this observation
				sensor = All_Observations[col]
				if ran:
					pred_state = random.randint(0,ROWS*COLS*HEADINGS-1)
				elif sensor_only:
					sensor_guess =  guess(sensor, All_Observations)
					pred_state = only_sensor_guess(sensor_guess)
				else:
					sensor_guess = guess(sensor, All_Observations) #Using the actual robot position, guess a reading and return the corresponding sensor data.
					f[:,i] = forward_algorithm(sensor_guess, Tr, f[:,i-1])
					pred_state = np.argmax(f[:,i])
				man_dist.append(distance(pred_state, rob.position))



				x1_temp = math.floor(pred_state / HEADINGS)
				x1 = math.floor(x1_temp / ROWS)
				y1 = x1_temp - ROWS * x1
				pred_board[x1,y1] = 2
				if (idx[0][0] == x1 and idx[0][1] == y1):
					correct_guesses += 1
				#print(pred_board) #Uncomment to print the movement of the guessed robot location
				#pred_board[x1,y1] = 0
				#print(rob.board) #Uncomment to print the movement of the robot location
				# time.sleep(0.3)

			
		#	print("Average Manhattan Distance after 100 steps: ",np.average(man_dist))
			tot_man_dist.append(man_dist)
		print("Total average for 100 loops: ", np.average(tot_man_dist))
		tot_man_dist_no_outliers = list(filter(lambda x: x !=0,[i for i in np.array(tot_man_dist).flatten() if i < 4])) #Somewhat arbitrary number
		if not ran and not sensor_only:
			print("Total average for 100 loops WITHOUT OUTLIERS: ", np.average(tot_man_dist_no_outliers))
		print("Percentage of correct guesses: ", correct_guesses/100, "%\n\n")




def guess(sensor, All_Observations):
	prob = np.zeros((ROWS,ROWS))
	prob = np.array([sensor[i] for i in range(0,len(sensor), HEADINGS)])
	prob = np.cumsum(prob)
	r = random.uniform(0, 1)
	for i in range(len(prob)):
		if r < prob[i]:
			idx = [math.floor(i / ROWS), i % COLS]
			col = idx[0] * ROWS + idx[1]
			return All_Observations[col]
	return All_Observations[-1]



def random_guess():
	return random.choice([(i,j) for i in range(ROWS) for j in range(COLS)])

def only_sensor_guess(sensor):
	prob = np.cumsum(sensor)
	r = random.uniform(0, 1)
	for i in range(len(prob)):
		if r < prob[i]:
			idx = [math.floor(i / ROWS), i % COLS]
			col = idx[0] * ROWS + idx[1]
			return col
	return None



def distance(state_1, state_2):
	x1_temp = math.floor(state_1 / HEADINGS)
	x1 = math.floor(x1_temp / ROWS)
	y1 = x1_temp - ROWS * x1
	return np.abs(x1 - state_2[0]) + np.abs(y1 - state_2[1])



def forward_algorithm(sensor, Tr, f): #There is an error here where np.linalg.norm fails
	if np.sum(sensor) != 0:
		f = np.diag(sensor) @ Tr.T @ f
		#alpha = 1/np.linalg.norm(f)
		#return alpha*f
	return f



class Robot:

	def __init__(self, position = [random.randint(0, ROWS - 1), random.randint(0, COLS - 1)], direction = random.randint(0,HEADINGS - 1), board = np.zeros((ROWS, COLS))):
		self.position = position
		self.direction = direction
		self.board = board
		self.board[self.position[0], self.position[1]] = 1


	def move_robot(self):
		self.board[self.position] = 0
		if is_Onboard(self.position[0] + directions[self.direction][0], self.position[1] + directions[self.direction][1]):
			if random.uniform(0, 1) < 0.7:
				self.position = [a + b for a, b in zip(self.position, directions[self.direction])]
				self.board[self.position[0], self.position[1]] = 1
				return

		available_directions = []
		for i in range(len(directions)):
			if is_Onboard(self.position[0] + directions[i][0], self.position[1] + directions[i][1]) and i != self.direction:
				available_directions.append(i)

		self.direction = random.choice(available_directions)
		self.position = [a + b for a, b in zip(self.position, directions[self.direction])]
		self.board[self.position[0], self.position[1]] = 1



	def obtain_sensor_reading(self):
		result = list(np.where(self.board == 1))
		idx = list(zip(result[0], result[1]))
		return idx


	def update_position_estimate(self):
		pass


def transition_matrix():
	trans_matrix = np.zeros((ROWS*COLS*HEADINGS, ROWS*COLS*HEADINGS)) #A matrix with the proababilities of moving from one state to another for all states.
	for i in range(len(trans_matrix)):
			direc = i % HEADINGS
			top_square =  - 32 + 0 - direc #Square to the top with upper direction
			right_square = 4 + 1 - direc #Square to the right with right direction
			bottom_square = 32 + 2 - direc #Square to the bottom with downwards direction
			left_square = - 4 + 3 - direc #Square to the left with left direction
			if upper_wall(i) and right_wall(i):
				no_reading_obs_matrix[i] = 0.625
				trans_matrix[i, i + left_square] = 0.5
				trans_matrix[i, i + bottom_square] = 0.5

			elif upper_wall(i) and left_wall(i):
				no_reading_obs_matrix[i] = 0.625
				trans_matrix[i, i + right_square] = 0.5
				trans_matrix[i, i + bottom_square] = 0.5

			elif bottom_wall(i) and right_wall(i):
				no_reading_obs_matrix[i] = 0.625
				trans_matrix[i, i + left_square] = 0.5
				trans_matrix[i, i + top_square] = 0.5

			elif bottom_wall(i) and left_wall(i):
				no_reading_obs_matrix[i] = 0.625
				trans_matrix[i, i + right_square] = 0.5
				trans_matrix[i, i + top_square] = 0.5

			elif upper_wall(i):
				no_reading_obs_matrix[i] = 0.5
				if direc == 0:
					trans_matrix[i, i + right_square] = 1/3
					trans_matrix[i, i + left_square] = 1/3
					trans_matrix[i, i + bottom_square] = 1/3
				elif direc == 1:
					trans_matrix[i, i + right_square] = 0.7
					trans_matrix[i, i + left_square] = 0.15
					trans_matrix[i, i + bottom_square] = 0.15
				elif direc == 3:
					trans_matrix[i, i + right_square] = 0.15
					trans_matrix[i, i + left_square] = 0.7
					trans_matrix[i, i + bottom_square] = 0.15

			elif right_wall(i):
				no_reading_obs_matrix[i] = 0.5
				if direc == 0:
					trans_matrix[i, i + top_square] = 0.7
					trans_matrix[i, i + left_square] = 0.15
					trans_matrix[i, i + bottom_square] = 0.15
				elif direc == 1:
					trans_matrix[i, i + top_square] = 1/3
					trans_matrix[i, i + left_square] = 1/3
					trans_matrix[i, i + bottom_square] = 1/3
				elif direc == 2:
					trans_matrix[i, i + top_square] = 0.15
					trans_matrix[i, i + left_square] = 0.15
					trans_matrix[i, i + bottom_square] = 0.7

			elif bottom_wall(i):
				no_reading_obs_matrix[i] = 0.5
				if direc == 3:
					trans_matrix[i, i + top_square] = 0.15
					trans_matrix[i, i + left_square] = 0.7
					trans_matrix[i, i + right_square] = 0.15
				elif direc == 1:
					trans_matrix[i, i + top_square] = 0.15
					trans_matrix[i, i + left_square] = 0.15
					trans_matrix[i, i + right_square] = 0.7
				elif direc == 2:
					trans_matrix[i, i + top_square] = 1/3
					trans_matrix[i, i + left_square] = 1/3
					trans_matrix[i, i + right_square] = 1/3

			elif left_wall(i):
				no_reading_obs_matrix[i] = 0.5
				if direc == 3:
					trans_matrix[i, i + top_square] = 1/3
					trans_matrix[i, i + bottom_square] = 1/3
					trans_matrix[i, i + right_square] = 1/3
				elif direc == 0:
					trans_matrix[i, i + top_square] = 0.7
					trans_matrix[i, i + bottom_square] = 0.15
					trans_matrix[i, i + right_square] = 0.15
				elif direc == 2:
					trans_matrix[i, i + top_square] = 0.15
					trans_matrix[i, i + bottom_square] = 0.7
					trans_matrix[i, i + right_square] = 0.15
				
			else:
				if direc == 0:
					trans_matrix[i, i + top_square] = 0.7
					trans_matrix[i, i + bottom_square] = 0.1
					trans_matrix[i, i + right_square] = 0.1
					trans_matrix[i, i + left_square] = 0.1
				elif direc == 1:
					trans_matrix[i, i + top_square] = 0.1
					trans_matrix[i, i + bottom_square] = 0.1
					trans_matrix[i, i + right_square] = 0.7
					trans_matrix[i, i + left_square] = 0.1
				elif direc == 2:
					trans_matrix[i, i + top_square] = 0.1
					trans_matrix[i, i + bottom_square] = 0.7
					trans_matrix[i, i + right_square] = 0.1
					trans_matrix[i, i + left_square] = 0.1
				else:
					trans_matrix[i, i + top_square] = 0.1
					trans_matrix[i, i + bottom_square] = 0.1
					trans_matrix[i, i + right_square] = 0.1
					trans_matrix[i, i + left_square] = 0.7

	return trans_matrix

			

def upper_wall(x):
	return x < ROWS*HEADINGS

def right_wall(x):
	for i in range(28, ROWS*COLS*HEADINGS, 32): #Change to 1 somewhere
		for j in range(HEADINGS):
			if (x == (i + j)):
				return True
	return False

def bottom_wall(x):
	return x > ROWS*COLS*HEADINGS - 33

def left_wall(x):
	for i in range(0, ROWS*COLS*HEADINGS, 32): #Change to 1 somewhere
		for j in range(HEADINGS):
			if (x == (i + j)):
				return True
	return False











def observation_matrix(x, y):
	neighbour_directions = [[1, 1], [1, 0],[1, -1],[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]
	second_neighbour_directions = [[2, 2], [2, 1], [2, 0], [2, -1], [2, -2], [1, -2], [0, -2], [-1, -2], 
	[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-1, 2], [0, 2], [1, 2]]

	neighbour_directions = [[1, 4], [1, 0],[1, -4],[0, -4], [-1, -4], [-1, 0], [-1, 4], [0, 4]]
	second_neighbour_directions = [[2, 8], [2, 4], [2, 0], [2, -4], [2, -8], [1, -8], [0, -8], [-1, -8], 
	[-2, -8], [-2, -4], [-2, 0], [-2, 4], [-2, 8], [-1, 8], [0, 8], [1, 8]]

	obser_matrix = np.zeros((ROWS,COLS*HEADINGS))
	y = y * 4
	for j in range(4):
		obser_matrix[x, y + j] = 0.1

	for i in range(len(neighbour_directions)):
		coord = [a + b for a, b in zip([x, y], neighbour_directions[i])]
		if is_Onboard2(coord[0], coord[1]):
			for j in range(4):
				obser_matrix[coord[0], coord[1] + j] = 0.05


	for i in range(len(second_neighbour_directions)):
		coord = [a + b for a, b in zip([x, y], second_neighbour_directions[i])]
		if is_Onboard2(coord[0], coord[1]):
			for j in range(4):
				obser_matrix[coord[0], coord[1] + j] = 0.025

	return obser_matrix
	
def build_observation_matrix():
	All_Observations = []
	for i in range(ROWS):
		for j in range(COLS):
			O = observation_matrix(i,j)
			obs_diagonal = np.array(O).flatten()
			All_Observations.append(obs_diagonal)
	All_Observations.append(no_reading_obs_matrix/np.linalg.norm(no_reading_obs_matrix))
	# All_Observations.append(np.ones(len(obs_diagonal))/256)
	return np.array(All_Observations)


				
			


def is_Onboard(x,y):
	return x >= 0 and x < ROWS and y >= 0 and y < COLS

def is_Onboard2(x,y):
	return x >= 0 and x < ROWS and y >= 0 and y < COLS*HEADINGS


main()
