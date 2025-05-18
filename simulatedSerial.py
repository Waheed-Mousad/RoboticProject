import random
import time

# Tunable constants for sensor behavior
DIST_NORMAL_PROB = 0.80  # 80% normal distance half if last move was not forward
DIST_CLOSE_PROB = 0.15   # 15% close obstacle double if last move was not forward
DIST_NEAR_PROB = 0.05    # 5% very near obstacle double if last move was not forward
DIST_NORMAL_RANGE = (100, 400)
DIST_CLOSE_RANGE = (50, 100)
DIST_NEAR_RANGE = (1, 50)
DIST_DECREASE_STEP = (10, 20)

IR_APPEAR_CHANCE = 0.20  # 20% chance when all white when last move was forward, double if last move was not forward
IR_PATTERN_WEIGHTS = {
    (1, 0, 0): 30,
    (0, 0, 1): 30,
    (0, 1, 0): 10,
    (1, 0, 1): 10,
    (0, 1, 1): 7,
    (1, 1, 0): 7,
    (1, 1, 1): 6
}

class SimulatedSerial:
    def __init__(self):
        print("⚠ Simulated serial initialized — no real Arduino connected.")
        self.last_ir = [0, 0, 0]
        self.last_distance = 300
        self.last_command = None
        self.last_command_last = None
        self.forward_counter = 0
        self.event_type = None
        self.event_entry = None
        self.event_progress = 0
        self.event_death_counter = 0
        self.event_goal = 0

    def write(self, data):
        command = data.decode('utf-8').strip()
        self.last_command = command


        if command in ('f', 'l', 'r'):
            #print(f"SimulatedSerial WRITE → {command} prve states → {self.last_ir}, {self.last_distance}", end=' ')
            self._update_ir_state(command)
            self._update_distance(command)
            #print(f"new states → {self.last_ir}, {self.last_distance}")
        # 's' does not change state


    def readline(self):
        data_str = f"{self.last_distance},{self.last_ir[0]},{self.last_ir[1]},{self.last_ir[2]}\n"
        time.sleep(0.05)
        return data_str.encode('utf-8')

    def reset_input_buffer(self):
        pass

    def reset(self):
        self.last_ir = [0, 0, 0]
        self.last_distance = 300
        self.last_command = None
        self.last_command_last = None
        self.forward_counter = 0
        self.event_type = None
        self.event_entry = None
        self.event_progress = 0
        self.event_death_counter = 0
        self.event_goal = 0
    def _update_ir_state(self, command):
        if self.event_type:
            self._process_event(command)
            return

        if any(self.last_ir):
            if self.last_ir == [1, 0, 1]:
                if command == 'f':
                    self.forward_counter += 1
                    if self.forward_counter >= 2:
                        self.last_ir = [0, 0, 0]
                        self.forward_counter = 0
                else:
                    self.last_ir = [0, 1, 0]
                    self.forward_counter = 0
                return

            if self.last_ir == [1, 1, 1]:
                if command == 'l':
                    self.last_ir = [0, 1, 1]
                elif command == 'r':
                    self.last_ir = [1, 1, 0]
                return

            if self.last_ir == [1, 0, 0]:
                if command == 'r':
                    self.last_ir = [0, 0, 0]
                elif command == 'l':
                    self.last_ir = [0, 1, 0]
            elif self.last_ir == [0, 0, 1]:
                if command == 'l':
                    self.last_ir = [0, 0, 0]
                elif command == 'r':
                    self.last_ir = [0, 1, 0]
            elif self.last_ir == [0, 1, 0]:
                if command == 'l':
                    self.last_ir = [0, 0, 1]
                elif command == 'r':
                    self.last_ir = [1, 0, 0]
            elif self.last_ir == [0, 1, 1]:
                if command == 'l':
                    self.last_ir = [0, 0, 1]
                elif command == 'r':
                    self.last_ir = [1, 1, 1]
            elif self.last_ir == [1, 1, 0]:
                if command == 'r':
                    self.last_ir = [1, 0, 0]
                elif command == 'l':
                    self.last_ir = [1, 1, 1]

            self.forward_counter = 0

        else:
            chance = IR_APPEAR_CHANCE
            if command != 'f':
                chance *= 2
                # === Chance to trigger an event ===
            if random.random() < 0.2:
                self.event_type = random.choice(["corridor", "uturn"])
                self.event_entry = "left" if random.random() < 0.5 else "right"
                self.event_progress = 0
                self.event_death_counter = 0
                self.event_goal = random.randint(5, 10)
                self.last_command_last = command
                self.last_ir = [1, 0, 0] if self.event_entry == "left" else [0, 0, 1]
            else:
                patterns = list(IR_PATTERN_WEIGHTS.keys())
                weights = list(IR_PATTERN_WEIGHTS.values())
                self.last_ir = list(random.choices(patterns, weights=weights, k=1)[0])

    def _process_event(self, command):
        print(f"Processing event: {self.event_type}, entry: {self.event_entry}, command: {command}, progress: {self.event_progress}, death counter: {self.event_death_counter}")
        if self.event_death_counter >= 3:
            print(f"Reached death condition :( death: {self.event_death_counter}")
            self.last_ir = [1, 1, 1]
            self.event_type = None
            return

        if self.event_type == "corridor":
            ir = self.last_ir

            # === Progress condition: safe forward movement ===
            if ir in ([0, 0, 1], [1, 0, 0], [1, 0, 1]) and command == 'f':
                self.event_progress += 1
                # IR may or may not improve after forward
                ran = random.random()
                if ran < 0.33:
                    self.last_ir = [1, 0, 0]
                elif ran < 0.66:
                    self.last_ir = [0, 0, 1]
                else:
                    self.last_ir = [1, 0, 1]
                self.event_death_counter = 0


            # === Recovery turns (fixing left/right walls) ===
            if ir == [1, 1, 0] and command == 'r':
                ran = random.random()
                if ran < 0.4:
                    self.last_ir = [1, 0, 1]
                elif ran < 0.8:
                    self.last_ir = [1, 0, 0]
                else:
                    self.last_ir = [0, 0, 1]
                self.event_death_counter = max(0, self.event_death_counter - 1)

            if ir == [0, 1, 1] and command == 'l':
                ran = random.random()
                if ran < 0.4:
                    self.last_ir = [1, 0, 1]
                elif ran < 0.8:
                    self.last_ir = [0, 0, 1]
                else:
                    self.last_ir = [1, 0, 0]
                self.event_death_counter = max(0, self.event_death_counter - 1)


            # === Dangerous wall-hugging turns ===
            if ir == [1, 0, 0] and command == 'l':
                ran = random.random()
                if ran < 0.8:
                    self.last_ir = [1, 1, 0]
                else:
                    self.last_ir = [0, 1, 0]
                self.event_death_counter += 1

            if ir == [0, 0, 1] and command == 'r':
                ran = random.random()
                if ran < 0.8:
                    self.last_ir = [0, 1, 1]
                else:
                    self.last_ir = [0, 1, 0]
                self.event_death_counter += 1

            if ir == [1, 0, 1] and command == 'l':
                self.last_ir = [1, 1, 0]  # left sensor worsens
                self.event_death_counter += 1

            elif ir == [1, 0, 1] and command == 'r':
                self.last_ir = [0, 1, 1]  # right sensor worsens
                self.event_death_counter += 1

            if ir == [1, 1, 0] and command in ('l', 'f'):
                self.last_ir = [1, 1, 1]
                self.event_death_counter += 1

            if ir == [0, 1, 1] and command in ('r', 'f'):
                self.last_ir = [1, 1, 1]
                self.event_death_counter += 1


            # === dangrous corrections ===
            if ir == [0, 1, 0] and command == 'r':
                if self.last_command_last == 'l':
                    ran = random.random()
                    if ran < 0.33:
                        self.last_ir = [0, 0, 1]
                    else:
                        self.last_ir = [0, 1, 1]
                else:
                    self.last_ir = [1, 1, 1]
                    self.event_death_counter += 1

            if ir == [0, 1, 0] and command == 'l':
                if self.last_command_last == 'r':
                    ran = random.random()
                    if ran < 0.33:
                        self.last_ir = [1, 0, 0]
                    else:
                        self.last_ir = [1, 1, 0]
                else:
                    self.last_ir = [1, 1, 1]
                    self.event_death_counter += 1

            if ir == [0, 1, 0] and command == 'f':
                if self.last_command_last == 'r':
                    ran = random.random()
                    if ran < 0.33:
                        self.last_ir = [1, 1, 1]
                    else:
                        self.last_ir = [0, 1, 1]
                    self.event_death_counter += 1
                elif self.last_command_last == 'l':
                    ran = random.random()
                    if ran < 0.33:
                        self.last_ir = [1, 1, 1]
                    else:
                        self.last_ir = [1, 1, 0]
                    self.event_death_counter += 1
                else:
                    self.last_ir = [1, 1, 1]
                    self.event_death_counter += 1

            if ir == [1, 1, 1]:
                if self.last_command_last == 'l':
                    if command == 'r':
                        self.last_ir = [1, 1, 0]
                        self.event_death_counter = max(0, self.event_death_counter - 1)
                    else:
                        self.last_ir = [1, 1, 1]
                        self.event_death_counter += 1

                elif self.last_command_last == 'r':
                    if command == 'l':
                        self.last_ir = [0, 1, 1]
                        self.event_death_counter = max(0, self.event_death_counter - 1)
                    else:
                        self.last_ir = [1, 1, 1]
                        self.event_death_counter += 1

                else:
                    if command == 'r':
                        self.last_ir = [1, 1, 0]
                        self.event_death_counter = max(0, self.event_death_counter - 1)
                    elif command == 'l':
                        self.last_ir = [0, 1, 1]
                        self.event_death_counter = max(0, self.event_death_counter - 1)
                    else:
                        self.last_ir = [1, 1, 1]
                        self.event_death_counter += 1

            if ir == [0, 0, 0]:
                if self.last_command == 'f':
                    ran = random.random()
                    if ran < 0.5:
                        self.last_ir = [0, 0, 0]
                    elif ran < 0.75:
                        self.last_ir = [1, 0, 0]
                    else:
                        self.last_ir = [0, 0, 1]
                    self.event_progress += 1
                if self.last_command == 'r':
                    self.last_ir = [0, 0, 1]
                if self.last_command == 'l':
                    self.last_ir = [1, 0, 0]

            # === Death threshold reached ===
            if self.event_death_counter >= 3:
                print(f"Reached death condition :( death: {self.event_death_counter}")
                self.last_ir = [1, 1, 1]
                self.event_type = None

            # === Successful completion ===
            if self.event_progress >= self.event_goal:
                print(f"event goal reached! {self.event_progress} > {self.event_goal}")
                self.last_ir = [0, 0, 0]
                self.event_type = None

            self.last_command_last = command
            return


        if self.event_type == "uturn":
            ir = self.last_ir
            cmd = command
            entry = self.event_entry

            # === Entry: LEFT ===
            if entry == "left":
                if ir == [1, 0, 0]:
                    if cmd in ('f', 'l'):
                        self.last_ir = [1, 1, 0]
                        self.event_death_counter += 1
                    elif cmd == 'r':
                        self.last_ir = [0, 0, 1]
                        self.event_death_counter = 0
                        self.event_progress += 1

                elif ir == [1, 1, 0]:
                    if cmd in ('f', 'l'):
                        self.last_ir = [1, 1, 1]
                        self.event_death_counter += 1
                    else:
                        self.last_ir = [0, 0, 1]
                        self.event_death_counter = max(0, self.event_death_counter - 1)

                elif ir == [1, 1, 1]:
                    if self.last_command_last == 'l':
                        if cmd in ('f', 'l'):
                            self.last_ir = [1, 1, 1]
                            self.event_death_counter += 1
                        else:
                            self.last_ir = [1, 1, 0]
                            self.event_death_counter = max(0, self.event_death_counter - 1)
                    else:
                        if cmd in ('f', 'r'):
                            self.last_ir = [1, 1, 1]
                            self.event_death_counter += 1
                        else:
                            self.last_ir = [0, 1, 1]
                            self.event_death_counter = max(0, self.event_death_counter - 1)

                elif ir == [0, 0, 1]:
                    if cmd == 'r':
                        self.last_ir = [0, 1, 1]
                        self.event_death_counter += 1
                    elif cmd == 'l':
                        self.last_ir = [1, 0, 0]
                        self.event_progress = max(0, self.event_progress - 1)
                    elif cmd == 'f':
                        self.last_ir = [1, 0, 0]
                        self.event_progress += 1
                        self.event_death_counter = 0

                elif ir == [0, 1, 1]:
                    if cmd == 'f':
                        self.last_ir = [1, 1, 1]
                        self.event_death_counter += 1
                    elif cmd == 'l':
                        self.last_ir = [0, 0, 1]
                        self.event_death_counter = max(0, self.event_death_counter - 1)
                    elif cmd == 'r':
                        self.last_ir = [1, 1, 1]
                        self.event_death_counter += 1

                elif ir == [0, 0, 0]:
                    if cmd == 'f':
                        self.last_ir = [0, 1, 0]
                    elif cmd == 'l':
                        self.last_ir = [1, 0, 0]
                    elif cmd == 'r':
                        self.last_ir = [0, 0, 1]

                elif ir == [0, 1, 0]:
                    if cmd == 'f':
                        self.last_ir = [1, 1, 0]
                        self.event_death_counter += 1
                    elif cmd == 'l':
                        self.last_ir = [1, 1, 0]
                        self.event_death_counter += 1
                    elif cmd == 'r':
                        self.last_ir = [0, 0, 1]
                        self.event_death_counter = 0

                elif ir == [1,0,1]:
                    if cmd == 'f':
                        ran = random.random()
                        if ran < 0.33:
                            self.last_ir = [1, 0, 1]
                            self.event_progress += 1
                        elif ran < 0.66:
                            self.last_ir = [0, 0, 1]
                            self.event_progress += 1
                        else:
                            self.last_ir = [1, 0, 0]
                            self.event_progress += 1



            # === Entry: RIGHT ===
            elif entry == "right":
                if ir == [0, 0, 1]:
                    if cmd in ('f', 'r'):
                        self.last_ir = [0, 1, 1]
                        self.event_death_counter += 1
                    elif cmd == 'l':
                        self.last_ir = [1, 0, 0]
                        self.event_death_counter = 0
                        self.event_progress += 1

                elif ir == [0, 1, 1]:
                    if cmd in ('f', 'r'):
                        self.last_ir = [1, 1, 1]
                        self.event_death_counter += 1
                    else:
                        self.last_ir = [0, 0, 1]
                        self.event_death_counter = max(0, self.event_death_counter - 1)

                elif ir == [1, 1, 1]:
                    if self.last_command_last == 'r':
                        if cmd in ('f', 'r'):
                            self.last_ir = [1, 1, 1]
                            self.event_death_counter += 1
                        else:
                            self.last_ir = [0, 1, 1]
                            self.event_death_counter = max(0, self.event_death_counter - 1)
                    else:
                        if cmd in ('f', 'l'):
                            self.last_ir = [1, 1, 1]
                            self.event_death_counter += 1
                        else:
                            self.last_ir = [1, 1, 0]
                            self.event_death_counter = max(0, self.event_death_counter - 1)

                elif ir == [1, 0, 0]:
                    if cmd == 'l':
                        self.last_ir = [1, 1, 0]
                        self.event_death_counter += 1
                    elif cmd == 'r':
                        self.last_ir = [0, 0, 1]
                        self.event_progress = max(0, self.event_progress - 1)
                    elif cmd == 'f':
                        self.last_ir = [0, 0, 1]
                        self.event_progress += 1

                elif ir == [1, 1, 0]:
                    if cmd == 'f':
                        self.last_ir = [1, 1, 1]
                        self.event_death_counter += 1
                    elif cmd == 'r':
                        self.last_ir = [0, 0, 1]
                        self.event_death_counter = max(0, self.event_death_counter - 1)
                    elif cmd == 'l':
                        self.last_ir = [1, 1, 1]
                        self.event_death_counter += 1

                elif ir == [0, 0, 0]:
                    if cmd == 'f':
                        self.last_ir = [0, 1, 0]
                    elif cmd == 'r':
                        self.last_ir = [0, 0, 1]
                    elif cmd == 'l':
                        self.last_ir = [1, 0, 0]

                elif ir == [0, 1, 0]:
                    if cmd == 'f':
                        self.last_ir = [0, 1, 1]
                        self.event_death_counter += 1
                    elif cmd == 'r':
                        self.last_ir = [0, 1, 1]
                        self.event_death_counter += 1
                    elif cmd == 'l':
                        self.last_ir = [1, 0, 0]
                        self.event_death_counter = 0

                elif ir == [1,0,1]:
                    if cmd == 'f':
                        ran = random.random()
                        if ran < 0.33:
                            self.last_ir = [1, 0, 1]
                            self.event_progress += 1
                        elif ran < 0.66:
                            self.last_ir = [0, 0, 1]
                            self.event_progress += 1
                        else:
                            self.last_ir = [1, 0, 0]
                            self.event_progress += 1

            # === Check for completion or failure ===
            if self.event_progress >= self.event_goal:
                print(f"event goal reached! {self.event_progress} > {self.event_goal}")
                self.event_type = None
                self.last_ir = [0, 0, 0]
            elif self.event_death_counter >= 3:
                print(f"Reached death condition :( death: {self.event_death_counter}")
                self.event_type = None
                self.last_ir = [1, 1, 1]

            self.last_command_last = command
            return

    def _update_distance(self, command):
        if command == 'f':
            if self.last_distance < 100:
                decrement = random.randint(*DIST_DECREASE_STEP)
                self.last_distance = max(1, self.last_distance - decrement)
            else:
                r = random.random()
                if r < DIST_NORMAL_PROB:
                    self.last_distance = random.randint(*DIST_NORMAL_RANGE)
                elif r < DIST_NORMAL_PROB + DIST_CLOSE_PROB:
                    self.last_distance = random.randint(*DIST_CLOSE_RANGE)
                else:
                    self.last_distance = random.randint(*DIST_NEAR_RANGE)
        elif command in ('l', 'r'):
            r = random.random()
            if r < DIST_NORMAL_PROB / 2:
                self.last_distance = random.randint(*DIST_NORMAL_RANGE)
            elif r < (DIST_NORMAL_PROB / 2 + DIST_CLOSE_PROB * 2) :
                self.last_distance = random.randint(*DIST_CLOSE_RANGE)
            else:
                self.last_distance = random.randint(*DIST_NEAR_RANGE)
