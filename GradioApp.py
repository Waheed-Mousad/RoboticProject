
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

BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
ser.reset_input_buffer()

# === Globals ===
latest_readings = {"distance": 0, "ir": [0, 0, 0]}
current_mode = "manual"  # manual, line, avoid

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

def decide_and_act():
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
        if dist < 30:
            send('s')
            time.sleep(0.2)
            send('b')
            time.sleep(0.4)
            if irR == 0:
                send('r')
            else:
                send('l')
            time.sleep(0.5)
            send('f')
        else:
            send('f')

# === Gradio Control Functions ===
def update():
    if current_mode in ["line", "avoid"]:
        decide_and_act()
    return latest_readings["distance"], *latest_readings["ir"]

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
        l = gr.Button("← Left")
        s = gr.Button("■ Stop")
        r = gr.Button("→ Right")
        b = gr.Button("↓ Back")

    with gr.Row():
        calL = gr.Slider(-100, 100, value=44, label="Left Calibration")
        calR = gr.Slider(-100, 100, value=-44, label="Right Calibration")

    manual.click(set_mode_manual, outputs=mode)
    line.click(set_mode_line, outputs=mode)
    avoid.click(set_mode_avoid, outputs=mode)

    f.click(lambda: manual_command('f'), outputs=[dist, irL, irM, irR])
    b.click(lambda: manual_command('b'), outputs=[dist, irL, irM, irR])
    l.click(lambda: manual_command('l'), outputs=[dist, irL, irM, irR])
    r.click(lambda: manual_command('r'), outputs=[dist, irL, irM, irR])
    s.click(lambda: manual_command('s'), outputs=[dist, irL, irM, irR])

    calL.change(lambda v: send_calibration(v, calR.value), inputs=calL)
    calR.change(lambda v: send_calibration(calL.value, v), inputs=calR)

    def update_ui():
        return update()

    timer = gr.Timer(0.5)
    timer.tick(fn=update_ui, outputs=[dist, irL, irM, irR])

app.launch(server_name="0.0.0.0", server_port=7860)
