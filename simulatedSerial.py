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
        self.last_distance = 100
        self.last_command = None
        self.forward_counter = 0

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

    def _update_ir_state(self, command):
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
            if random.random() < chance:
                patterns = list(IR_PATTERN_WEIGHTS.keys())
                weights = list(IR_PATTERN_WEIGHTS.values())
                self.last_ir = list(random.choices(patterns, weights=weights, k=1)[0])

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
