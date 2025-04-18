# Smart Car Raspberry Pi Control Script with Full Logic + Gradio UI

import serial
import threading
import time
import gradio as gr
import serial.tools.list_ports
# Try to auto-detect serial port
ports = list(serial.tools.list_ports.comports())
SERIAL_PORT = None
for p in ports:
    if "Arduino" in p.description or "ttyACM" in p.device or "USB" in p.device:
        SERIAL_PORT = p.device
        break

if SERIAL_PORT is None:
    raise Exception("Arduino not found. Please connect the device.")

BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
ser.reset_input_buffer()
scan_trigger_distance = 40
resume_forward_distance = 80
turning_threshhold = 1
# === Globals ===
latest_readings = {"distance": 0, "ir": [0, 0, 0]}
current_mode = "manual"  # manual, line, avoid
latest_mode = "manual"
thread = False
turn_to_left = True
# === Serial Read Thread ===
def read_serial():
    global latest_readings
    while True:

        try:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == 4:
                latest_readings = {
                    "distance": int(parts[0]),
                    "ir": [int(parts[1]), int(parts[2]), int(parts[3])]
                }
        except:
            continue

threading.Thread(target=read_serial, daemon=True).start()

# === Arduino Command Sender ===
def send(cmd):
    ser.write(cmd.encode('utf-8'))
    time.sleep(0.05)

def send_calibration(left, right):
    send(f"L{left}\n")
    send(f"R{right}\n")

def send_speed(speed):
    send(f"S{speed}\n")

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

# === Gradio UI ===
with gr.Blocks() as app:
    gr.Markdown("# Smart Car Control Panel")

    with gr.Row():
        mode = gr.Textbox(label="Current Mode", value="manual")
        dist = gr.Number(label="Distance (cm)")
        irL = gr.Number(label="IR Left")
        irM = gr.Number(label="IR Middle")
        irR = gr.Number(label="IR Right")

    with gr.Row():
        manual = gr.Button("Manual Mode")
        line = gr.Button("Line-Follow Mode")
        avoid = gr.Button("Obstacle-Avoid Mode")

    with gr.Row():
        f = gr.Button("↑ Forward")
        b = gr.Button("↓ Back")
        l = gr.Button("← Left")
        r = gr.Button("→ Right")
        s = gr.Button("■ Stop")

    with gr.Row():
        calL = gr.Slider(minimum=-100, maximum=155, value=0, label="Left Calibration")
        calR = gr.Slider(minimum=-100, maximum=155, value=0, label="Right Calibration")

    with gr.Row():
        Speed = gr.Slider(minimum=0, maximum=255, value=100, label="Speed")

    with gr.Row(visible=False) as avoid_controls:
        scan_slider = gr.Slider(minimum=0, maximum=100, value=scan_trigger_distance, step=1, label="Scan Trigger Distance")
        resume_slider = gr.Slider(minimum=0, maximum=200, value=resume_forward_distance, step=1, label="Forward Resume Distance")

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
    timer.tick(fn=update_ui, outputs=[mode,dist, irL, irM, irR])

app.launch(server_name="0.0.0.0", server_port=7860)
