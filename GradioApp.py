# Smart Car Raspberry Pi Control Script with Full Logic + Gradio UI
"""
real envoirment test result
batter than expected
simulation need to improve
it get stuck at corners for repeated actions = panalise
still prefer to repeat useless action = panalise
sometimes likes to move forward no matter what = reduce reward
maybe add reading thread for ml, make aurdino stops at sensor change,
take new step at sensor change?

"""
import serial
import threading
import time
import gradio as gr
import serial.tools.list_ports
import os
from model import QTrainer, Linear_QNet
import model
print(model.__file__)

import numpy as np
import random
import torch
from collections import deque
from simulatedSerial import *
# === Serial Setup ===
ports = list(serial.tools.list_ports.comports())
SERIAL_PORT = None
for p in ports:
    if "Arduino" in p.description or "ttyACM" in p.device or "USB" in p.device:
        SERIAL_PORT = p.device
        break

BAUD_RATE = 9600

if SERIAL_PORT is None:
    print("⚠ Arduino not found. Switching to simulated serial.")
    ser = SimulatedSerial()
else:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()

# === Model Path Setup ===
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pth')
# === Globals ===
# Please spare my life for spaghetti I will make is moduler later trust me bro
#TODO Make it modular Trust me bro

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class CarAgent:
    def __init__(self, model):
        self.n_game = 0
        self.extra_games = 0
        self.epsilon = 0  # exploration randomness
        self.gamma = 0.9  # discount rate
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
        self.epsilon = 80 - self.n_game - self.extra_games # decrease randomness over time
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
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
agent = None
total_score = 0
scan_trigger_distance = 40
resume_forward_distance = 80
turning_threshhold = 1
latest_readings = {"distance": 0, "ir": [0, 0, 0]}
current_mode = "manual"  # manual, line, avoid
latest_mode = "manual"
thread = False
turn_to_left = True
NORMAL = False
reading_thread = None
SIMULATION = False
# === Serial Read Thread ===
def update_reading_thread():
    global latest_readings
    while NORMAL:
        read_serial()
        time.sleep(0.05)

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



def start_normal_mode():
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
    prev_readings = latest_readings.copy()
    send('f')
    start = time.time()
    while time.time() - start < 2.0:
        read_serial()
        if latest_readings != prev_readings:
            break
        time.sleep(0.005)
    send('s')


def ML_backward():
    prev_readings = latest_readings.copy()
    send('b')
    start = time.time()
    while time.time() - start < 2.0:
        read_serial()
        if latest_readings != prev_readings:
            break
        time.sleep(0.005)
    send('s')

def ML_left():
    prev_readings = latest_readings.copy()
    send('l')
    start = time.time()
    while time.time() - start < 0.5:
        read_serial()
        if latest_readings != prev_readings:
            break
        time.sleep(0.005)
    send('s')

def ML_right():
    prev_readings = latest_readings.copy()
    send('r')
    start = time.time()
    while time.time() - start < 0.5:
        read_serial()
        if latest_readings != prev_readings:
            break
        time.sleep(0.005)
    send('s')
# === Normal mod functions ===
def avoid_obstacle():
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

def update_scan_thresholds(scan_val, resume_val):
    global scan_trigger_distance, resume_forward_distance
    # Enforce resume > scan
    if resume_val <= scan_val:
        resume_val = scan_val + 1  # ensure it's strictly greater
    scan_trigger_distance = scan_val
    resume_forward_distance = resume_val
    return scan_val, resume_val

# === Gradio Control Functions ===
def update():
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
    global current_mode
    current_mode = "manual"
    send('s')
    return "manual"

def set_mode_line():
    global current_mode
    current_mode = "line"
    send('s')
    return "line"

def set_mode_avoid():
    global current_mode
    current_mode = "avoid"
    send('s')
    return "avoid"

def manual_command(cmd):
    print(f"manual command: {cmd}")
    global current_mode
    current_mode = "manual"
    send(cmd)
    return update()

# === ML Functions ===

def load_or_create_agent():
    global agent
    load = False
    model = Linear_QNet(8, 256, 3)
    if os.path.exists(model_path):
        model.load_model(file_name='model.pth')
        text = "✅ Loaded existing model."
        load = True
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(file_name='model.pth')
        text = "🆕 Created and saved new model."
        load = False

    agent = CarAgent(model)
    if load:
        agent.extra_games = 70
    return text

def delete_model():
    global agent
    if os.path.exists(model_path):
        os.remove(model_path)
        agent = None
        return "✅ Model deleted."
    else:
        return "❌ No model found to delete."

def map_distance_onehot(distance):
    if distance < 10:
        category = 0  # very close
    elif distance < 30:
        category = 1  # close
    elif distance < 100:
        category = 2  # far
    else:
        category = 3  # very far

    one_hot = [0, 0, 0, 0]
    one_hot[category] = 1
    return one_hot

def get_state_from_car(latest_readings):
    read_serial() # read the latest data
    distance_onehot = map_distance_onehot(latest_readings["distance"])
    ir_left = latest_readings["ir"][0]
    ir_middle = latest_readings["ir"][1]
    ir_right = latest_readings["ir"][2]

    state = distance_onehot + [ir_left, ir_middle, ir_right]
    return np.array(state, dtype=int)

DIST_REWARD_MATRIX = [
    [-30, 10, 20, 20],  # very close
    [-10, -10, 5, 5],  # close
    [-0.5, -0.5, 0.5, 0.5],  # far
    [-0.5, -0.5, 0.5, 0.5],  # very far
]

IR_REWARD_MATRIX = [
    [0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
    [5, -5, -10, -15, 0, -10, -15, -20],
    [10, 5, -10, -15, 5, 0, -15, -20],
    [15, 5, 5, -15, 5, 0, -15, -20],
    [5, 0, -10, -15, -5, -10, -15, -20],
    [10, 0, 0, -10, 5, -10, -10, -20],
    [15, 5, 5, -15, 5, 0, -15, -20],
    [20, 15, 10, 5, 15, 10, 5, -20]
]

def compute_reward(prev_state, next_state, action_taken, paused):
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
    print(
        f"prev_state: {prev_state}, next_state: {next_state}, action_taken: {action_taken}, reward: {reward}, paused: {paused}")

    return reward

def execute_action(action):
    if action[0] == 1:
        ML_forward()
    elif action[1] == 1:
        ML_left()
    elif action[2] == 1:
        ML_right()

def toggle_pause():
    global ML_PAUSED
    ML_PAUSED = not ML_PAUSED
    if ML_PAUSED:
        return "Training paused. Reposition the robot."
    else:
        return "Training resumed. Starting new episode."


def stop_training():
    global ML_RUNNING
    ML_RUNNING = False
    return "Training stopped."

def start_training():
    global ML_RUNNING, ML_PAUSED, agent, total_score

    if agent is None:
        print("❌ Error: Agent not loaded. Please load or create the agent first.")
        return "❌ Agent not loaded. Please load or create the agent first."
    state_old = np.zeros(8)
    state_new = np.zeros(8)
    ML_RUNNING = True
    ML_PAUSED = False
    total_score = 0
    MAX_STEPS = 100
    current_steps = 0
    episode_score = 0
    print("🚀 Training started.")
    yield "🚀 Training started."
    while ML_RUNNING:
        if ((state_old[5] == 1 or state_old[0] == 1) and action[0] == 1) and SIMULATION is True:
            ML_PAUSED = True
            current_steps = -1

        state_old = np.append(get_state_from_car(latest_readings), int(ML_PAUSED))
        action = agent.get_action(state_old)
        execute_action(action)
        time.sleep(0.01)
        read_serial()  # ← this ensures latest_readings updates from the simulator plz work
        state_new = np.append(get_state_from_car(latest_readings), int(ML_PAUSED))
        current_steps += 1
        print(f"current_steps: {current_steps} - ", end="")
        # stop the episode if reach the max steps as well
        if current_steps >= MAX_STEPS and SIMULATION is True and ML_PAUSED is False:
            ML_PAUSED = True
            current_steps = 0

        if ML_PAUSED:
            reward = compute_reward(state_old, state_new, action, paused=True)
            done = True
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)
            episode_score += reward
            agent.n_game += 1
            total_score += episode_score
            avg_score = total_score / agent.n_game

            agent.train_long_memory()
            agent.model.save()

            message = f"Training paused. Reposition the robot. ✅ Episode {agent.n_game} saved. episode Score: {episode_score}, Avg Score: {avg_score:.2f}"
            print(message)
            yield message
            episode_score = 0
            while ML_PAUSED and ML_RUNNING:
                time.sleep(0.1)
                if SIMULATION is True:
                    time.sleep(0.5)
                    ML_PAUSED = False
            continue

        reward = compute_reward(state_old, state_new, action, paused=False)

        episode_score += reward
        done = False

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

    print("🛑 Training stopped.")
    yield "🛑 Training stopped."

def set_extra_games(n):
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
        gr.Markdown("ML Mode Tab")
        gr.Markdown("🚧 *Future machine learning features will go here.*")
        ml_init = gr.Button("Initialize ML Mode ")
        with gr.Row(visible=False) as ml_mode_controls:
            # test for ML commands buttons (forward, left, right)
            ml_f = gr.Button("↑ Forward")
            ml_l = gr.Button("← Left")
            ml_r = gr.Button("→ Right")
            load_model_btn = gr.Button("Load or Create agent")
            delete_model_btn = gr.Button("Delete Model")


        with gr.Row(visible=False) as ml_mode_loaded_model:
            start_btn = gr.Button("Start Training")
            pause_btn = gr.Button("Pause / Resume Episode")
            stop_btn = gr.Button("Stop Training")
            extra_games_input = gr.Number(label="Extra Games", value=0)
            set_extra_btn = gr.Button("Set Extra Games")
            safeguard_btn = gr.Button("📦 Safeguard Save")

        status_text = gr.Textbox(label="Training Status", value="Idle", interactive=False)



        gr.Markdown("### Machine Learning Controls")



        # === Bindings ===
        ml_f.click(lambda: ML_forward(), outputs=[])
        ml_l.click(lambda: ML_left(), outputs=[])
        ml_r.click(lambda: ML_right(), outputs=[])

        load_model_btn.click(fn=load_or_create_agent, outputs=status_text)
        load_model_btn.click(lambda: gr.update(visible=True), outputs=ml_mode_loaded_model)
        delete_model_btn.click(fn=delete_model, outputs=status_text)
        delete_model_btn.click(lambda: gr.update(visible=False), outputs=ml_mode_loaded_model)

        start_btn.click(fn=start_training, outputs=status_text)
        pause_btn.click(fn=toggle_pause, outputs=status_text)
        stop_btn.click(fn=stop_training, outputs=status_text)
        set_extra_btn.click(set_extra_games, inputs=extra_games_input, outputs=status_text)
        safeguard_btn.click(safeguard_save, outputs=status_text)

    # === Gradio UI Update ===
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


app.launch(server_name="0.0.0.0", server_port=7860)

