# Smart Car Raspberry Pi Control Script with Full Logic + Gradio UI
"""

"""
import serial
import threading
import time
import gradio as gr
import serial.tools.list_ports
import os
from model import QTrainer, Linear_QNet
import model
from collections import deque



print(model.__file__)

import numpy as np
import random
import torch
from collections import deque
from simulatedSerial import *
import csv
import uuid

LOG_FOLDER = os.path.join(os.path.dirname(__file__), 'training_logs')
os.makedirs(LOG_FOLDER, exist_ok=True)

# === Serial Setup ===
ports = list(serial.tools.list_ports.comports())
SERIAL_PORT = None
for p in ports: #scan all serial ports and find the Aurdino
    if "Arduino" in p.description or "ttyACM" in p.device or "USB" in p.device:
        SERIAL_PORT = p.device
        break

BAUD_RATE = 9600
SIMULATION = False
if SERIAL_PORT is None: # if no serial found, start simulation
    print("⚠ Arduino not found. Switching to simulated serial.")
    ser = SimulatedSerial()
    SIMULATION = True
    print("simulation is:", SIMULATION)
else:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()

# === Model Path Setup ===
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pth')
# === Globals ===
# Please spare my life for spaghetti I will make is moduler later trust me bro
#TODO Make it modular Trust me bro
"""
 dear reader, I know I said I will make is moduler, however when I wrote this code only I, allah and github co-pilot
 knew how it worked, now only allah the all-mighty know, I made a huge mistake of using tenth of global variables, and
 alot of dependency between functions, when ever I try to make it moduler it breaks. basically what I am trying to say, 
 the code is almost 900 lines of code and almost unreadable, bring coffe before trying to read it, actually bring 2
"""
sensor_history = deque(maxlen=3) # keep the last 3 sensor reading, this is for ML related training
action_history = deque(maxlen=3)  # Keep the last 3 actions, this is for ML related training
MAX_MEMORY = 100_000 # max memory size for agent
BATCH_SIZE = 1000 # how many memory to learn in one go
LR = 0.001 # learning rate

# this is the agent class, it the agent. for ML, reinforcement learning to be specific
# this is mostly taken from
# Goswami, V. (n.d.). SnakeGameAI [Computer software]. GitHub. https://github.com/vedantgoswami/SnakeGameAI
# with small changes
class CarAgent:
    def __init__(self, model):
        self.n_game = 0
        self.extra_games = 0
        self.epsilon = 0  # exploration randomness
        self.gamma = 0.95  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = model
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 200 - self.n_game - self.extra_games # decrease randomness over time
        final_move = [0, 0, 0]
        if self.extra_games > 999: # no randomness at all if extra games is greater than 999
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        elif random.randint(0, 300) <  max(20, self.epsilon):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

ML_RUNNING = False  # controls overall training loop
ML_PAUSED = False   # controls pause/resume between episodes
agent = None #by defualt agent is none.
total_score = 0 # total score the agent has gained over all episodes
scan_trigger_distance = 40 # I don't remember what is this
resume_forward_distance = 80 # I don't remember what is this
turning_threshhold = 1 # I don't remember what is this
latest_readings = {"distance": 0, "ir": [0, 0, 0]} #this is latest reading of sensors
current_mode = "manual"  # manual, line, avoid this is for normal mode
latest_mode = "manual" # again for normal mode
thread = False # this will allow the thread (I believe for contsant reading in normal mode) to keep looping
turn_to_left = True # I don't remember what is this
NORMAL = False # is it normal mode or ML mode?
reading_thread = None # actually this is for constant reading thread in ML mod, not sure what the previous was
# I think for the modes in normal mode
score_history = deque(maxlen=50) # save the last 50 episodes score

# === Serial Read Thread ===
def update_reading_thread():
    global latest_readings
    while NORMAL: #only run if in normal mode
        read_serial()
        time.sleep(0.05)

# === Read serial and update latest reading ===
def read_serial():
    global latest_readings
    try:
        line = None
        ser.reset_input_buffer()
        while line is None or line == '':
            line = ser.readline().decode('utf-8').strip()

        parts = line.split(',')
        if len(parts) == 4:
            latest_readings = {
            "distance": int(parts[0]),
                "ir": [int(parts[1]), int(parts[2]), int(parts[3])]
            }
    except:
        print("Error reading serial data")
        return


# === initialise mode ===
def start_normal_mode():
    # this will make ML flags false and make normal flag true, and finally start constant reading thread
    global NORMAL, reading_thread, ML_RUNNING, ML_PAUSED
    # stop ML mode if active
    ML_RUNNING = False
    ML_PAUSED = False

    if not NORMAL:
        NORMAL = True
        if reading_thread is None or not reading_thread.is_alive():
            reading_thread = threading.Thread(target=update_reading_thread, daemon=True)
            reading_thread.start()

def start_ml_mode():
    global NORMAL, thread
    # basically make normal flag False and close any other normal related thread
    NORMAL = False     # stop reading thread
    thread = False     # stop avoid_obstacle or avoid_line threads

# === Arduino Command Sender ===
def send(cmd):
    ser.write(cmd.encode('utf-8'))
    time.sleep(0.05)

def send_calibration(left, right):
    send(f"L{left}\n")
    send(f"R{right}\n")

def send_speed(speed):
    send(f"S{speed}\n")
# === ML mod functions     ===
def ML_forward():
    # move forward, stop if time out or sensor changed
    prev_readings = get_state_from_car()
    send('f')
    start = time.time()
    while time.time() - start < 2.0:

        if not np.array_equal(get_state_from_car(), prev_readings):
            break
        time.sleep(0.005)
    send('s')

def ML_backward():
    # move backward, stop if time out or sensor changed note: this is not used
    prev_readings = get_state_from_car()
    send('b')
    start = time.time()
    while time.time() - start < 2.0:

        if not np.array_equal(get_state_from_car(), prev_readings):
            break
        time.sleep(0.001)
    send('s')

def ML_left():
    # turn left, stop if time out or sensor changed
    prev_readings = get_state_from_car()
    send('l')
    start = time.time()
    while time.time() - start < 0.5:

        if not (np.array_equal(get_state_from_car(), prev_readings)) and (time.time() - start< 0.1):
            break
        time.sleep(0.001)
    send('s')

def ML_right():
    # turn right, stop if time out or sensor changed
    prev_readings = get_state_from_car()
    send('r')
    start = time.time()
    time.sleep(0.5)
    while time.time() - start < 0.5:

        if not (np.array_equal(get_state_from_car(), prev_readings)) and (time.time() - start< 0.1):
            break
        time.sleep(0.001)
    send('s')

# === Normal mod functions ===
def avoid_obstacle():
    """
    constantly check if distance sensor reading is lower then threshold
    if lower, turn to the left, until freed, if not for freed for a time
    next turn is opposite direction
    :return:
    """
    global turn_to_left
    print("obstacle mode")
    while thread:
        dist = latest_readings["distance"]
        if dist < scan_trigger_distance:
            send('s')
            time.sleep(0.05)
            start = time.time()
            if turn_to_left:
                send('l')
            else:
                send('r')

            while latest_readings["distance"] < resume_forward_distance or (time.time() - start) < 0.5:
                time.sleep(0.01)


            if (time.time() - start) > 2:
                #reverse turn to left
                turn_to_left = not turn_to_left


        send('f')
        time.sleep(0.05)
    print("exiting obstacle avoidance thread")

def avoid_line():
    """
    | IRleft | IRmiddle | IRright | Decision |
    |--------|----------|---------|----------|
    |   0    |    0     |    0    | forward  |
    |   0    |    0     |    1    | left     |
    |   0    |    1     |    0    | left     |
    |   0    |    1     |    1    | left     |
    |   1    |    0     |    0    | right    |
    |   1    |    0     |    1    | forward  |
    |   1    |    1     |    0    | right    |
    |   1    |    1     |    1    | left     |
    :return:
    """
    print("line mode")
    while thread:
        if latest_readings["ir"][0] == 0 and latest_readings["ir"][1] == 0 and latest_readings["ir"][2] == 0:
            send('f')
        elif latest_readings["ir"][0] == 1 and latest_readings["ir"][1] == 0 and latest_readings["ir"][2] == 0:
            send('r')
        elif latest_readings["ir"][0] == 0 and latest_readings["ir"][1] == 0 and latest_readings["ir"][2] == 1:
            send('l')
        elif latest_readings["ir"][0] == 1 and latest_readings["ir"][1] == 0 and latest_readings["ir"][2] == 1:
            send('f')
        elif latest_readings["ir"][0] == 1 and latest_readings["ir"][1] == 1 and latest_readings["ir"][2] == 0:
            send('r')
        elif latest_readings["ir"][0] == 0 and latest_readings["ir"][1] == 1 and latest_readings["ir"][2] == 1:
            send('l')
        elif latest_readings["ir"][0] == 1 and latest_readings["ir"][1] == 1 and latest_readings["ir"][2] == 1:
            send('l')
        elif latest_readings["ir"][0] == 0 and latest_readings["ir"][1] == 1 and latest_readings["ir"][2] == 0:
            send('l')
        time.sleep(0.05)
        
    print("exiting line avoidance thread")

# this function will update the scans threshold for obstacle avoidance mode
def update_scan_thresholds(scan_val, resume_val):
    global scan_trigger_distance, resume_forward_distance
    # Enforce resume > scan
    if resume_val <= scan_val:
        resume_val = scan_val + 1  # ensure it's strictly greater
    scan_trigger_distance = scan_val
    resume_forward_distance = resume_val
    return scan_val, resume_val

# === Gradio Control Functions for normal mode ===
def update():
    # this functon update the UI with latest reading and information, it is tied to a timer
    global current_mode
    global latest_mode
    global thread
    if current_mode != latest_mode:
        print("mode changed from:", latest_mode, "to ", current_mode)
        latest_mode = current_mode
        if latest_mode == "avoid":
            thread = False
            time.sleep(2)
            thread = True
            threading.Thread(target=avoid_obstacle, daemon=True).start()
        if latest_mode == "line":
            thread = False
            time.sleep(2)
            thread = True
            threading.Thread(target=avoid_line, daemon=True).start()
        if latest_mode == "manual":
            thread = False
    return current_mode, latest_readings["distance"], *latest_readings["ir"]

def set_mode_manual():
    # set the mode to manual
    global current_mode
    current_mode = "manual"
    send('s')
    return "manual"

def set_mode_line():
    # set the mode to line following
    global current_mode
    current_mode = "line"
    send('s')
    return "line"

def set_mode_avoid():
    # set the mode to obstacle avoidance3
    global current_mode
    current_mode = "avoid"
    send('s')
    return "avoid"

def manual_command(cmd):
    # change mode to manual if not already manual
    # then send the command to the Arduino
    print(f"manual command: {cmd}")
    global current_mode
    current_mode = "manual"
    send(cmd)
    return update()

# === ML Functions ===
def list_available_models():
    """
    list the available models for the drop down menu
    :return:
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_dir, exist_ok=True)
    return [f for f in os.listdir(model_dir) if f.endswith('.pth')]

def load_selected_model(filename):
    """
    load the selected model to the agent.
    :param filename:
    :return:
    """
    global agent
    full_path = os.path.join(os.path.dirname(__file__), 'model', filename)
    model = Linear_QNet(28, 256, 3)
    if os.path.exists(full_path):
        model.load_model(file_name=filename)
        agent = CarAgent(model)
        agent.extra_games = 0
        return f"✅ Loaded model: {filename}"
    else:
        return "❌ Model not found."

def create_new_model(filename):
    """
    create a new model, read the name from the text box
    :param filename:
    :return:
    """
    global agent
    if not filename.endswith('.pth'):
        filename += '.pth'
    full_path = os.path.join(os.path.dirname(__file__), 'model', filename)
    model = Linear_QNet(28, 256, 3)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    model.save(file_name=filename)
    agent = CarAgent(model)
    return f"🆕 Created and saved new model: {filename}"

def delete_model(filename):
    """
    delete model
    :param filename:
    :return:
    """
    global agent
    if not filename.endswith('.pth'):
        filename += '.pth'
    full_path = os.path.join(os.path.dirname(__file__), 'model', filename)
    if os.path.exists(full_path):
        os.remove(full_path)
        if agent and hasattr(agent.model, 'file_name') and agent.model.file_name == filename:
            agent = None
        return f"✅ Deleted model: {filename}"
    else:
        return "❌ Model not found."

def map_distance_onehot(distance):
    """
    turn the distance from value between 0 and 400
    to one hot encoded array
    example: 70 to [0, 0, 1, 0]
    :param distance:
    :return: one hot encoded array for distance
    """
    if distance < 10:
        category = 0  # very close
    elif distance < 20:
        category = 1  # close
    elif distance < 100:
        category = 2  # far
    else:
        category = 3  # very far

    one_hot = [0, 0, 0, 0]
    one_hot[category] = 1
    return one_hot

def get_state_from_car():
    """
    read the sensor value, one hot encoded distance and IR sensor reading
    :return: numpy 1D array
    """
    read_serial() # read the latest data
    distance_onehot = map_distance_onehot(latest_readings["distance"])
    ir_left = latest_readings["ir"][0]
    ir_middle = latest_readings["ir"][1]
    ir_right = latest_readings["ir"][2]

    state = distance_onehot + [ir_left, ir_middle, ir_right]
    return np.array(state, dtype=int)
"""
# distance reward matrix

             | 0,0,0,1 | 0,0,1,0 | 0,1,0,0 | 1,0,0,0
-------------|---------|---------|---------|---------
0,0,0,1      |   0.5   |   0.5   |   0     |   0
0,0,1,0      |   0.5   |   0.5   |   0     |   0
0,1,0,0      |   5     |   5     | -10     | -10
1,0,0,0      |  10     |  10     |  10     | -50

the matrix is reversed from the previous table
"""

DIST_REWARD_MATRIX = [
    [-50, 10, 20, 20],  # very close
    [-10, -10, 5, 5],  # close
    [-0.5, -0.5, 0.5, 0.5],  # far
    [-0.5, -0.5, 0.5, 0.5],  # very far
]

"""
IR reward matrix, give reward on previous sensors values and next sensor values

           | 0,0,0 | 0,0,1 | 0,1,0 | 0,1,1 | 1,0,0 | 1,0,1 | 1,1,0 | 1,1,1
-----------|-------|-------|-------|-------|-------|-------|-------|-------
0,0,0      |   5   | -0.5  | -0.5  | -0.5  | -0.5  | -0.5  | -0.5  | -0.5
0,0,1      |  20   |   0   | -10   | -10   |   0   |   0   | -10   | -30
0,1,0      |  20   |   5   | -10   | -10   |   5   |   5   | -10   | -30
0,1,1      |  20   |   5   | -10   | -10   |   5   |   5   | -10   | -30
1,0,0      |  20   |   0   | -10   | -10   |   0   |   0   | -10   | -30
1,0,1      |  20   |   0   | -10   | -10   |   0   |   0   | -10   | -30
1,1,0      |  20   |   5   | -10   | -10   |   5   |   5   | -10   | -30
1,1,1      |  20   |  10   |  10   |  10   |  10   |  10   |  10   | -30

"""
IR_REWARD_MATRIX = [
    [5,-0.5, -0.5,-0.5,-0.5,-0.5, -0.5, -0.5],
    [20,  -0.5,  -10, -10,   -0.5,   -0.5,  -15, -30],
    [20,  5,  -10, -10,   5,   5,  -15, -30],
    [20,  5,  -10, -10,   5,   5,  -15, -30],
    [20,  -0.5,  -10, -10,   -0.5,   -0.5,  -15, -30],
    [20,  -0.5,  -10, -10,   -0.5,   -0.5,  -10, -30],
    [25,  5,  -10, -10,   5,   5,  -15, -30],
    [20, 10,   10,  10,  10,  10,   10, -30]
]

def compute_reward(prev_state, next_state, action_taken, paused):
    """
    Calculates the reward signal based on state transitions and action taken.

    :param prev_state: The agent's previous state vector, a list of 14 elements
                       (4 distance one-hot + 3 IR + 2 previous actions + 2 previous sensor states + paused flag)
    :param next_state: The agent's new state after taking the action
    :param action_taken: One-hot encoded list representing the action [forward, left, right]
    :param paused: Boolean indicating whether training is currently paused
    :return: A numeric reward value reflecting progress, penalties, or reinforcement
    """
    # turn state into index
    # the state is a list of 7 elements like this [0 0 0 1 0 0 1]
    # last 3 are ir values left middle right
    # first 4 are distance values
    # 0 = very close, 1 = close, 2 = far, 3 = very far
    # 0 = left, 1 = middle, 2 = right
    prev_ir = prev_state[4:7]
    next_ir = next_state[4:7]
    prev_distance = prev_state[0:4]
    next_distance = next_state[0:4]
    # turn ir into index
    prev_ir_index = prev_ir[0] * 4 + prev_ir[1] * 2 + prev_ir[2]
    next_ir_index = next_ir[0] * 4 + next_ir[1] * 2 + next_ir[2]
    # turn distance into index
    prev_distance_index = prev_distance[0] * 0 + prev_distance[1] * 1 + prev_distance[2] * 2 + prev_distance[3] * 3
    next_distance_index = next_distance[0] * 0 + next_distance[1] * 1 + next_distance[2] * 2 + next_distance[3] * 3

    # get the reward from the matrix
    distance_reward = DIST_REWARD_MATRIX[prev_distance_index][next_distance_index]
    ir_reward = IR_REWARD_MATRIX[prev_ir_index][next_ir_index]

    # get the action taken
    # not implemented

    # return the reward
    if paused:
        reward = distance_reward + ir_reward - 10
    else:
        reward = distance_reward + ir_reward
    # if previous ir were zeros and action was forward
    if np.all(prev_ir == 0) and action_taken[0] == 1 and (prev_distance[3] == 1 or prev_distance[2] == 1):
        reward += 2


    # === Repetition penalty ===
    recent = list(action_history)

    # Penalize if all 3 recent actions are left
    if len(recent) == 3 and all(a == "l" for a in recent):
        reward -= 5  # adjust penalty as needed

    # Penalize if all 3 recent actions are right
    if len(recent) == 3 and all(a == "r" for a in recent):
        reward -= 5  # adjust penalty as needed

    # Penalize if only turning (left or right) and no forward
    if len(recent) == 3 and all(a in ("l", "r") for a in recent) and "f" not in recent:
        reward -= 10  # stronger penalty for being stuck turning

    print(
        f"prev_state: {prev_state}, next_state: {next_state}, action_taken: {action_taken}, reward: {reward}, paused: {paused}")
    return reward

# action encoding history
ACTION_ENCODING = {
    "f": [1, 0, 0],
    "l": [0, 1, 0],
    "r": [0, 0, 1],
}

def extend_state_with_history(state, history, paused_flag):
    """
    this function will extend the 7 states input and make them into 28 state input
    that included the current states, the previous 2 states and 2 actions as well as a paused flag
    :param state:
    :param history:
    :param paused_flag:
    :return:
    """
    # Store only the sensor part (distance + IR) from current state into history
    sensor_history.append(state[:7])  # 4 distance one-hot + 3 IR

    # Last 2 actions
    recent_actions = list(history)[-2:]
    while len(recent_actions) < 2:
        recent_actions.insert(0, None)
    encoded_actions = [ACTION_ENCODING[a] if a else [0, 0, 0] for a in recent_actions]
    flat_actions = [x for trio in encoded_actions for x in trio]

    # Last 2 sensor states (excluding current one)
    recent_sensors = list(sensor_history)[-3:-1]
    while len(recent_sensors) < 2:
        recent_sensors.insert(0, [0] * 7)
    flat_sensors = [x for s in recent_sensors for x in s]

    return np.append(state, flat_actions + flat_sensors + [int(paused_flag)])


def execute_action(action):
    # execute action based on what recieved, also save in the action history que
    global action_history
    if action[0] == 1:
        action_history.append("f")
        ML_forward()
    elif action[1] == 1:
        action_history.append("l")
        ML_left()
    elif action[2] == 1:
        action_history.append("r")
        ML_right()


def toggle_pause():
    # reverse the paused flag
    global ML_PAUSED
    ML_PAUSED = not ML_PAUSED
    if ML_PAUSED:
        return "Training paused. Reposition the robot."
    else:
        return "Training resumed. Starting new episode."


def stop_training():
    # stop training (and inference)
    global ML_RUNNING
    ML_RUNNING = False
    return "Training stopped."

def start_training():
    """
    Starts the reinforcement learning training loop for the smart car agent.

    This function controls one full training lifecycle:
    - Initializes training logs and flags
    - Repeatedly interacts with the environment by collecting state, selecting an action,
      executing it, and observing the result
    - Computes reward from state transitions and stores experience in memory
    - Triggers short- and long-term training for the model
    - Periodically pauses the episode to allow repositioning or resetting the robot
    - Saves progress, including model weights and training metrics, to a CSV log file

    Yields:
        str: Status updates such as training start, pause, and episode summaries.
    """
    global ML_RUNNING, ML_PAUSED, agent, total_score
    log_filename = f"log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    LOG_PATH = os.path.join(LOG_FOLDER, log_filename)

    with open(LOG_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "episode",
            "episode_score",
            "avg_score",
            "steps_this_episode",
            "score_per_step"
        ])
    if agent is None:
        print("❌ Error: Agent not loaded. Please load or create the agent first.")
        return "❌ Agent not loaded. Please load or create the agent first."
    state_old = np.zeros(14)
    state_new = np.zeros(14)
    ML_RUNNING = True
    ML_PAUSED = False
    total_score = 0
    MAX_STEPS = 128
    current_steps = 0
    current_steps_2 = 0
    episode_score = 0
    print("🚀 Training started.")
    yield "🚀 Training started."
    while ML_RUNNING:
        if (state_old[0] == 1 and state_old[4] == 1 and state_old[5] == 1 and state_old[6] == 1 and action[
            0] == 1) and SIMULATION is True:
            ML_PAUSED = True
            current_steps = -1

        state_old = extend_state_with_history(get_state_from_car(), action_history, ML_PAUSED)
        action = agent.get_action(state_old)
        execute_action(action)
        time.sleep(0.01)
        read_serial()  # ← this ensures latest_readings updates from the simulator plz work
        state_new = extend_state_with_history(get_state_from_car(), action_history, ML_PAUSED)
        current_steps += 1
        print(f"current_steps: {current_steps} - ", end="")
        # stop the episode if reach the max steps as well
        if current_steps >= MAX_STEPS and SIMULATION is True and ML_PAUSED is False:
            ML_PAUSED = True
            current_steps_2 = current_steps
            current_steps = 0

        if ML_PAUSED:
            if current_steps != 0:
                reward = compute_reward(state_old, state_new, action, paused=True)
                done = True
                agent.train_short_memory(state_old, action, reward, state_new, done)
                agent.remember(state_old, action, reward, state_new, done)
                episode_score += reward
            else:
                # Don't penalize or reward on timeout
                done = True

            # Always do these:
            agent.n_game += 1
            score_history.append(episode_score)
            avg_score = sum(score_history) / len(score_history)

            agent.train_long_memory()
            agent.model.save()


            message = f"Training paused. Reposition the robot. ✅ Episode {agent.n_game} saved. episode Score: {episode_score}, Avg Score: {avg_score:.2f}"
            print(message)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            steps_this_episode = current_steps_2 if current_steps_2 > 0 else 1  # prevent div by zero
            score_per_step = episode_score / steps_this_episode

            with open(LOG_PATH, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    timestamp,
                    agent.n_game,
                    episode_score,
                    avg_score,
                    steps_this_episode,
                    score_per_step
                ])

            yield message

            episode_score = 0
            while ML_PAUSED and ML_RUNNING:

                time.sleep(0.1)
                if SIMULATION is True:
                    time.sleep(0.5)
                    ML_PAUSED = False

            # Clear history after pause ends (i.e., before new episode)
            ser.reset()
            sensor_history.clear()
            action_history.clear()
            continue

        reward = compute_reward(state_old, state_new, action, paused=False)

        episode_score += reward
        done = False

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

    print("🛑 Training stopped.")
    yield "🛑 Training stopped."

def start_inference():
    """
    the same as training function, just does not train memory and does not save logs
    :return:
    """
    global ML_RUNNING, ML_PAUSED, agent

    if agent is None:
        print("❌ Error: Agent not loaded. Please load or create the agent first.")
        return "❌ Agent not loaded. Please load or create the agent first."

    agent.extra_games = 1000  # Disable randomness
    ML_RUNNING = True
    ML_PAUSED = False
    MAX_STEPS = 100
    current_steps = 0
    episode_score = 0
    print("🚗 Inference started (no training).")
    yield "🚗 Inference started. Running model at full capacity."

    while ML_RUNNING:
        state_old = extend_state_with_history(get_state_from_car(), action_history, ML_PAUSED)
        action = agent.get_action(state_old)
        execute_action(action)
        time.sleep(0.01)
        read_serial()
        state_new = extend_state_with_history(get_state_from_car(), action_history, ML_PAUSED)

        reward = compute_reward(state_old, state_new, action, paused=False)
        episode_score += reward
        current_steps += 1
        print(f"Step {current_steps}: reward = {reward:.2f}, total = {episode_score:.2f}")

        # Auto-pause condition (e.g., episode complete)
        if current_steps >= MAX_STEPS:
            ML_PAUSED = True
            current_steps = 0

        if ML_PAUSED:
            print(f"⏸️ Paused. Episode total reward: {episode_score:.2f}")
            yield f"⏸️ Paused. Episode total reward: {episode_score:.2f}"

            episode_score = 0
            # Wait until unpaused
            while ML_PAUSED and ML_RUNNING:
                ser.reset()
                time.sleep(0.1)


            # Clear history before next episode
            sensor_history.clear()
            action_history.clear()
            continue

    print("🛑 Inference stopped.")
    yield "🛑 Inference stopped."

def set_extra_games(n):
    """
    set extra games to control epsilon value
    :param n:
    :return:
    """
    if agent:
        agent.extra_games = int(n)
        return f"✅ extra_games set to {n}"
    else:
        return "❌ Agent not loaded."

def safeguard_save():
    if agent:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"safeguard_{timestamp}.pth"
        full_path = os.path.join(os.path.dirname(model_path), filename)
        agent.model.save(file_name=full_path)
        return f"✅ Model saved as {filename}"
    return "❌ Agent not loaded."

def start_training_button():
    """
    start training
    :return:
    """
    global ML_RUNNING

    if ML_RUNNING:
        return "⚠ Training is already running!"

    training_thread = threading.Thread(target=start_training)
    training_thread.start()
    return "🚀 Training loop started in background."
# === Gradio UI ===
with gr.Blocks() as app:
    gr.Markdown("# Smart Car Control Panel")

    with gr.Tab("Normal"):
        # === Gradio UI: Normal Mode Tab ===
        # This tab provides manual and rule-based control options for the smart car.
        # - Users can initialize normal mode and view live sensor data (distance + IR).
        # - Includes buttons to switch between manual, line-follow, and obstacle-avoid modes.
        # - Manual driving controls (↑ ↓ ← → Stop) send direct commands to the car.
        # - Calibration sliders adjust motor offsets for left/right wheels.
        # - Speed control and obstacle thresholds are adjustable via sliders.
        # - The interface updates every 0.5s to reflect the current state of the car.
        gr.Markdown("Normal Mode Tab")
        normal_init = gr.Button("Initialize Normal Mode ")
        with gr.Row(visible=False) as normal_mode_controls_Row1:
            mode = gr.Textbox(label="Current Mode", value="manual")
            dist = gr.Number(label="Distance (cm)")
            irL = gr.Number(label="IR Left")
            irM = gr.Number(label="IR Middle")
            irR = gr.Number(label="IR Right")

        with gr.Row(visible=False) as normal_mode_controls_Row2:
            manual = gr.Button("Manual Mode")
            line = gr.Button("Line-Follow Mode")
            avoid = gr.Button("Obstacle-Avoid Mode")

        with gr.Row(visible=False) as normal_mode_controls_Row3:
            f = gr.Button("↑ Forward")
            b = gr.Button("↓ Back")
            l = gr.Button("← Left")
            r = gr.Button("→ Right")
            s = gr.Button("■ Stop")

        with gr.Row(visible=False) as normal_mode_controls_Row4:
            calL = gr.Slider(minimum=-100, maximum=155, value=0, label="Left Calibration")
            calR = gr.Slider(minimum=-100, maximum=155, value=0, label="Right Calibration")

        with gr.Row(visible=False) as normal_mode_controls_Row5:
            Speed = gr.Slider(minimum=0, maximum=255, value=150, label="Speed")

        with gr.Row(visible=False) as avoid_controls:
            scan_slider = gr.Slider(minimum=0, maximum=100, value=scan_trigger_distance, step=1, label="Scan Trigger Distance")
            resume_slider = gr.Slider(minimum=0, maximum=200, value=resume_forward_distance, step=1, label="Forward Resume Distance")

        # === Bindings ===

        manual.click(set_mode_manual, outputs=mode)
        line.click(set_mode_line, outputs=mode)
        avoid.click(set_mode_avoid, outputs=mode)

        f.click(lambda: manual_command('f'), outputs=[mode, dist, irL, irM, irR])
        b.click(lambda: manual_command('b'), outputs=[mode, dist, irL, irM, irR])
        l.click(lambda: manual_command('l'), outputs=[mode, dist, irL, irM, irR])
        r.click(lambda: manual_command('r'), outputs=[mode, dist, irL, irM, irR])
        s.click(lambda: manual_command('s'), outputs=[mode, dist, irL, irM, irR])

        calL.change(lambda v: send_calibration(v, calR.value), inputs=calL)
        calR.change(lambda v: send_calibration(calL.value, v), inputs=calR)
        Speed.change(lambda v: send_speed(v), inputs=Speed)

        scan_slider.change(update_scan_thresholds, inputs=[scan_slider, resume_slider],
                           outputs=[scan_slider, resume_slider])
        resume_slider.change(update_scan_thresholds, inputs=[scan_slider, resume_slider],
                             outputs=[scan_slider, resume_slider])

        mode.change(lambda m: gr.update(visible=(m == "avoid")), inputs=mode, outputs=avoid_controls)

        def update_ui():
            return update()

        timer = gr.Timer(0.5)
        timer.tick(fn=update_ui, outputs=[mode, dist, irL, irM, irR])

    with gr.Tab("ML"):
        # === Gradio UI: ML Mode Tab ===
        # This tab provides controls for managing the machine learning model used by the smart car.
        # - Users can initialize ML mode and interact with the agent via forward/left/right buttons.
        # - A dropdown lists available saved models for selection and loading.
        # - Users can create new models, delete existing ones, and refresh the model list.
        # - Once a model is loaded, users can start training, pause/resume episodes, or run inference.
        # - Extra training parameters like "extra_games" can be set.
        # - A safeguard button allows saving the current model with a timestamped filename.
        # - Training and inference progress is shown through a status text box.
        gr.Markdown("ML Mode Tab")
        gr.Markdown("🚧 *Future machine learning features will go here.*")
        ml_init = gr.Button("Initialize ML Mode ")
        with gr.Row(visible=False) as ml_mode_controls:
            # test for ML commands buttons (forward, left, right)
            ml_f = gr.Button("↑ Forward")
            ml_l = gr.Button("← Left")
            ml_r = gr.Button("→ Right")
            model_dropdown = gr.Dropdown(choices=list_available_models(), label="Select Model", interactive=True)
            refresh_models_btn = gr.Button("🔄 Refresh Models List")
            create_model_text = gr.Textbox(label="New Model Name (e.g., my_model.pth)")
            create_model_btn = gr.Button("Create New Model")
            load_model_btn = gr.Button("Load Selected Model")
            delete_model_btn = gr.Button("Delete Model")


        with gr.Row(visible=False) as ml_mode_loaded_model:
            start_btn = gr.Button("Start Training")
            pause_btn = gr.Button("Pause / Resume Episode")
            stop_btn = gr.Button("Stop Training")
            start_infer_btn = gr.Button("Run Inference Only")
            extra_games_input = gr.Number(label="Extra Games", value=0)
            set_extra_btn = gr.Button("Set Extra Games")
            safeguard_btn = gr.Button("📦 Safeguard Save")

        status_text = gr.Textbox(label="Training Status", value="Idle", interactive=False)



        gr.Markdown("### Machine Learning Controls")



        # === Bindings ===
        ml_f.click(lambda: ML_forward(), outputs=[])
        ml_l.click(lambda: ML_left(), outputs=[])
        ml_r.click(lambda: ML_right(), outputs=[])

        refresh_models_btn.click(fn=lambda: gr.update(choices=list_available_models()), outputs=model_dropdown)
        load_model_btn.click(fn=load_selected_model, inputs=model_dropdown, outputs=status_text)
        create_model_btn.click(fn=create_new_model, inputs=create_model_text, outputs=status_text)

        load_model_btn.click(lambda: gr.update(visible=True), outputs=ml_mode_loaded_model)
        delete_model_btn.click(fn=delete_model, inputs=model_dropdown, outputs=status_text)
        delete_model_btn.click(lambda: gr.update(visible=False), outputs=ml_mode_loaded_model)

        start_btn.click(fn=start_training, outputs=status_text)
        pause_btn.click(fn=toggle_pause, outputs=status_text)
        stop_btn.click(fn=stop_training, outputs=status_text)
        set_extra_btn.click(set_extra_games, inputs=extra_games_input, outputs=status_text)
        safeguard_btn.click(safeguard_save, outputs=status_text)
        start_infer_btn.click(fn=start_inference, outputs=status_text)

    # === Gradio UI Update ===
    # for some reason those needed to be outside
    normal_init.click(lambda: gr.update(visible=True), outputs=normal_mode_controls_Row1)
    normal_init.click(lambda: gr.update(visible=True), outputs=normal_mode_controls_Row2)
    normal_init.click(lambda: gr.update(visible=True), outputs=normal_mode_controls_Row3)
    normal_init.click(lambda: gr.update(visible=True), outputs=normal_mode_controls_Row4)
    normal_init.click(lambda: gr.update(visible=True), outputs=normal_mode_controls_Row5)
    normal_init.click(lambda: gr.update(visible=False), outputs=ml_mode_controls)
    normal_init.click(lambda: gr.update(visible=False), outputs=ml_mode_loaded_model)
    normal_init.click(start_normal_mode, outputs=[])

    ml_init.click(lambda: gr.update(visible=True), outputs=ml_mode_controls)
    ml_init.click(lambda: gr.update(visible=False), outputs=normal_mode_controls_Row1)
    ml_init.click(lambda: gr.update(visible=False), outputs=normal_mode_controls_Row2)
    ml_init.click(lambda: gr.update(visible=False), outputs=normal_mode_controls_Row3)
    ml_init.click(lambda: gr.update(visible=False), outputs=normal_mode_controls_Row4)
    ml_init.click(lambda: gr.update(visible=False), outputs=normal_mode_controls_Row5)
    ml_init.click(start_ml_mode, outputs=[])

# lunch the UI
app.launch(server_name="0.0.0.0", server_port=7860)

