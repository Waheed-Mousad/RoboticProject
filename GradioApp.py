# Smart Car Raspberry Pi Control Script with Full Logic + Gradio UI

import serial
import threading
import time
import gradio as gr
import serial.tools.list_ports
import os
from model import QTrainer, Linear_QNet
import numpy as np
import random
import torch
from collections import deque
# Try to auto-detect serial port
ports = list(serial.tools.list_ports.comports())
SERIAL_PORT = None
for p in ports:
    if "Arduino" in p.description or "ttyACM" in p.device or "USB" in p.device:
        SERIAL_PORT = p.device
        break

if SERIAL_PORT is None:
    raise Exception("Arduino not found. Please connect the device.")
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pth')
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
ser.reset_input_buffer()
# === Globals ===
# Please spare my life for spaghetti I will make is moduler later trust me bro
#TODO Make it modular Trust me bro

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class CarAgent:
    def __init__(self, model):
        self.n_game = 0
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
        self.epsilon = 80 - self.n_game  # decrease randomness over time
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
    send('f')
    time.sleep(1)
    send('s')

def ML_backward():
    send('b')
    time.sleep(1)
    send('s')

def ML_left():
    send('l')
    time.sleep(0.5)
    send('s')

def ML_right():
    send('r')
    time.sleep(0.5)
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
    model = Linear_QNet(7, 256, 3)
    if os.path.exists(model_path):
        model.load(file_name='model.pth')
        text = "✅ Loaded existing model."
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(file_name='model.pth')
        text = "🆕 Created and saved new model."

    agent = CarAgent(model)
    return text


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

def compute_reward(prev_state, next_state, paused):
    # === Reward Parameters List ===
    # [0] BASE_FORWARD_REWARD
    # [1] OBSTACLE_VERY_CLOSE_PENALTY
    # [2] ALL_WHITE_REWARD
    # [3] SIDE_IR_BLACK_PENALTY
    # [4] MIDDLE_IR_BLACK_PENALTY
    # [5] ALL_IR_BLACK_PENALTY
    # [6] RECOVERY_SIDE_REWARD
    # [7] RECOVERY_MIDDLE_REWARD
    # [8] MANUAL_PAUSE_PENALTY
    reward_values = [
        1,    # BASE_FORWARD_REWARD
        -10,  # OBSTACLE_VERY_CLOSE_PENALTY
        5,    # ALL_WHITE_REWARD
        -5,   # SIDE_IR_BLACK_PENALTY
        -20,  # MIDDLE_IR_BLACK_PENALTY
        -50,  # ALL_IR_BLACK_PENALTY
        10,   # RECOVERY_SIDE_REWARD
        20,   # RECOVERY_MIDDLE_REWARD
        -100  # MANUAL_PAUSE_PENALTY
    ]

    reward = reward_values[0]  # BASE_FORWARD_REWARD

    distance_onehot = next_state[:4]
    very_close = distance_onehot[0]

    prev_ir_left = prev_state[4]
    prev_ir_middle = prev_state[5]
    prev_ir_right = prev_state[6]

    curr_ir_left = next_state[4]
    curr_ir_middle = next_state[5]
    curr_ir_right = next_state[6]

    if very_close == 1:
        reward += reward_values[1]  # OBSTACLE_VERY_CLOSE_PENALTY

    if curr_ir_left == 0 and curr_ir_middle == 0 and curr_ir_right == 0:
        reward += reward_values[2]  # ALL_WHITE_REWARD

    if curr_ir_left == 1 or curr_ir_right == 1:
        reward += reward_values[3]  # SIDE_IR_BLACK_PENALTY

    if curr_ir_middle == 1:
        reward += reward_values[4]  # MIDDLE_IR_BLACK_PENALTY

    if curr_ir_left == 1 and curr_ir_middle == 1 and curr_ir_right == 1:
        reward += reward_values[5]  # ALL_IR_BLACK_PENALTY

    if (prev_ir_left == 1 and curr_ir_left == 0) or (prev_ir_right == 1 and curr_ir_right == 0):
        reward += reward_values[6]  # RECOVERY_SIDE_REWARD

    if prev_ir_middle == 1 and curr_ir_middle == 0:
        reward += reward_values[7]  # RECOVERY_MIDDLE_REWARD

    if paused:
        reward += reward_values[8]  # MANUAL_PAUSE_PENALTY

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
    global ML_RUNNING, ML_PAUSED, agent

    if agent is None:
        print("❌ Error: Agent not loaded. Please load or create the agent first.")
        return "❌ Agent not loaded. Please load or create the agent first."

    ML_RUNNING = True
    ML_PAUSED = False

    print("🚀 Training started.")

    while ML_RUNNING:
        # === Main training step ===
        state_old = get_state_from_car(latest_readings)
        action = agent.get_action(state_old)
        execute_action(action)

        time.sleep(0.1)  # allow robot to respond + sensors to update

        state_new = get_state_from_car(latest_readings)

        if ML_PAUSED: # if paused, save the episode and wait for resume
            reward = compute_reward(state_old, state_new, paused=True)
            done = True
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

            agent.n_game += 1
            agent.train_long_memory()
            agent.model.save()
            print(f"✅ Episode {agent.n_game} completed and saved (paused).")

            while ML_PAUSED and ML_RUNNING:
                time.sleep(0.1)
            continue  # restart loop

        reward = compute_reward(state_old, state_new, paused=False)

        done = False  # no auto-ending yet and will never be added

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

    print("🛑 Training stopped.")
    return "🛑 Training stopped."

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
            Speed = gr.Slider(minimum=0, maximum=255, value=100, label="Speed")

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
            load_model_btn = gr.Button("Load or Create agent")


        with gr.Row(visible=False) as ml_mode_loaded_model:
            start_btn = gr.Button("Start Training")
            pause_btn = gr.Button("Pause / Resume Episode")
            stop_btn = gr.Button("Stop Training")

        status_text = gr.Textbox(label="Training Status", value="Idle", interactive=False)



        gr.Markdown("### Machine Learning Controls")



        # === Bindings ===



        load_model_btn.click(fn=load_or_create_agent, outputs=status_text)
        load_model_btn.click(lambda: gr.update(visible=True), outputs=ml_mode_loaded_model)

        start_btn.click(fn=start_training_button, outputs=status_text)
        pause_btn.click(fn=toggle_pause, outputs=status_text)
        stop_btn.click(fn=stop_training, outputs=status_text)

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

