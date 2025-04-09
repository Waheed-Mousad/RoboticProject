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
scanning_event = threading.Event()
scanning_event.clear()
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
ser.reset_input_buffer()
scan_trigger_distance = 40
resume_forward_distance = 80
# === Globals ===
latest_readings = {"distance": 0, "ir": [0, 0, 0]}
current_mode = "manual"  # manual, line, avoid

# === Serial Read Thread ===
def read_serial():
    global latest_readings
    while True:
        while not scanning_event.is_set():
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

def decide_and_act():
    global current_mode
    scanning_event.clear()
    dist = latest_readings["distance"]
    irL, irM, irR = latest_readings["ir"]

    if current_mode == "line":
        if irM:
            send('f')
        elif irL:
            send('l')
        elif irR:
            send('r')
        else:
            send('s')


    elif current_mode == "avoid":

        # === MINOR UPDATE: New area scan logic ===
        print("obstacle mode")
        if dist <= scan_trigger_distance:
            scanning_event.set()
            print("Obstacle detected, performing area scan")
            send('s')  # Stop immediately
            time.sleep(0.05)
            send('A')  # Ask Arduino to perform area scan
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            time.sleep(2.5)  # Give Arduino time to scan and send back values
            # Read scan values from serial (expects: M,R,L)

            try:
                start = time.time()
                print("Waiting for scan values...")
                while time.time() - start < 2.0:
                    if current_mode == "manual" or current_mode == "line":
                        scanning_event.clear()
                        return
                    line = ser.readline().decode('utf-8').strip()
                    print(f"Received line: {line}")
                    if line.startswith('SCAN:'):
                        _, values = line.split(":")
                        m, r, l = map(int, values.split(","))
                        print(f"Scan values: M={m}, R={r}, L={l}")
                        if r > l:
                            send('r')
                        else:
                            send('l')
                        start = time.time()
                        while latest_readings["distance"] < resume_forward_distance and time.time() - start < 2.0:
                            scanning_event.clear()
                            if current_mode == "manual" or current_mode == "line":
                                return
                            time.sleep(0.05)

                        send('s')
                        time.sleep(0.05)
                        send('f')


                        break  # stop the wait loop after handling SCAN
            except:
                print("Error reading scan values")
                pass

        else:
            send('f')
            time.sleep(0.05)

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
    if current_mode in ["line", "avoid"]:
        decide_and_act()
    return current_mode, latest_readings["distance"], *latest_readings["ir"]

def set_mode_manual():
    print("manual mode")
    global current_mode
    scanning_event.clear()
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
    scanning_event.clear()
    print(scanning_event.is_set())
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
        l = gr.Button("← Left")
        s = gr.Button("■ Stop")
        r = gr.Button("→ Right")
        b = gr.Button("↓ Back")

    with gr.Row():
        calL = gr.Slider(-100, 100, value=44, label="Left Calibration")
        calR = gr.Slider(-100, 100, value=-44, label="Right Calibration")

    with gr.Row(visible=False) as avoid_controls:
        scan_slider = gr.Slider(minimum=0, maximum=100, value=40, step=1, label="Scan Trigger Distance")
        resume_slider = gr.Slider(minimum=0, maximum=200, value=80, step=1, label="Forward Resume Distance")

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
