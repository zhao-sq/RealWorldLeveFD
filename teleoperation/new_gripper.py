import time
import serial
from random import randrange
import pyspacemouse

class New_Gripper():
    def __init__(self, ser, spacemouse_rotate_deg=120, warmup_time=1, operation_time=0.01):
        """Initialize gripper.

        Args:
            ser: Serial port.
            gripper_operation_time (time): Time for gripper to open or close.
        """
        self.FULLY_OPEN_DEG = 1872
        self.FULLY_CLOSE_DEG = 0
        self.SPEED_RAD_PER_S = 120.0
        self.SPACEMOUSE_ROTATE_DEG = spacemouse_rotate_deg
        self.ser = ser
        self.operation_time = operation_time
        self.warmup_time = warmup_time
        for _ in range(16):
            self.send_command("open", self.SPEED_RAD_PER_S)
            time.sleep(0.1)
        self.data_last_sent_time = time.time()
        self.pos_deg = 1872 # we start with the gripper fully open

    def fully_open(self):
        for _ in range(16):
            self.send_command("open", self.SPEED_RAD_PER_S)
            time.sleep(0.1)
        self.pos_deg = 1872
        self.data_last_sent_time = time.time()

    def fully_close(self):
        for _ in range(16):
            self.send_command("close", self.SPEED_RAD_PER_S)
            time.sleep(0.1)
        self.pos_deg = 0
        self.data_last_sent_time = time.time()

    def get_ser(self):
        """
        Returns:
            ser: Serial port.
        """
        return self.ser
    
    def get_operation_time(self):
        """
        Returns:
            self.operation_time (time): Time for gripper to open or close.
        """
        return self.operation_time
        
    def get_position_deg(self):
        """
        Returns:
            state (int): The current position degree of the gripper. 0 means fully close and 1872 means fully open. 
        """
        return self.pos_deg
    
    def get_data_last_sent_time(self):
        """
        Returns:
            self.data_last_sent_time (time): The time we last operate the gripper.
        """
        return self.data_last_sent_time

    def is_in_action(self):
        """Check if gripper is still in action.

        Args:
            time_after_data_last_sent: The time we sent data to gripper.

        Returns:
            boolean: True if gripper is still in action and False if gripper is not in action.
        """
        time_passed_after_data_last_sent = time.time() - self.get_data_last_sent_time()
        if time_passed_after_data_last_sent >= self.get_operation_time():
            return False
        else:
            return True
        
    def send_data(self, spacemouse_state):
        """Check spacemouse_state and move gripper accordingly.

        Args:
            spacemouse_state: The current state of the spacemouse.
        """
        if spacemouse_state.buttons[-1]:
            if spacemouse_state.y > 0:
                move_direction = 'close'
                print(f"move_direction: {move_direction}")
                self.move(move_direction, self.SPACEMOUSE_ROTATE_DEG)
            elif spacemouse_state.y < 0:
                move_direction = 'open'
                print(f"move_direction: {move_direction}")
                self.move(move_direction, self.SPACEMOUSE_ROTATE_DEG)

    
    def move(self, move_direction, rotate_deg):
        current_pos_deg = self.get_position_deg()
        if move_direction == "open":
            max_open_deg = self.FULLY_OPEN_DEG - current_pos_deg
            rotate_deg = min(rotate_deg, max_open_deg)
            self.pos_deg = self.pos_deg + rotate_deg
        elif move_direction == "close":
            max_close_deg = current_pos_deg - self.FULLY_CLOSE_DEG
            rotate_deg = min(rotate_deg, max_close_deg)
            self.pos_deg = self.pos_deg - rotate_deg
        else:
            raise Exception("Invalid move_direction.")
        print(f"Rotate degree: {rotate_deg}")
        self.send_command(move_direction, rotate_deg)
        self.data_last_sent_time = time.time()
        print(f"Curent position degree: {self.pos_deg}")

    def move_to_pos(self, position_deg):
        current_deg = self.get_position_deg()
        if position_deg < current_deg:
            self.move('close', current_deg - position_deg)
        elif position_deg > current_deg:
            self.move('open', position_deg - current_deg)
        
    def send_command(self, move_direction, position_deg):
        serial_port = self.get_ser()
        ser = serial.Serial(serial_port, baudrate=115200, timeout=1)
        
        frame = self.build_frame(move_direction=move_direction, position_deg=position_deg, speed_rad_per_s=self.SPEED_RAD_PER_S)
        # print("发送数据帧:", frame.hex(' ').upper())

        ser.write(frame)
        ser.close()
    
    def build_frame(
        self,
        move_direction, # "close" or "open"
        position_deg,     # 目标角度，单位为度
        speed_rad_per_s   # 目标速度，单位为 rad/s
    ):
        frame = [0x7B]  # 帧头
        device_addr=0x01
        control_mode=0x02
        
        if move_direction == "close":
            direction=0x01
        else:
            direction=0x00
            
        subdivision=0x20      # 32细分
        frame.append(device_addr)
        frame.append(control_mode)
        frame.append(direction)
        frame.append(subdivision)

        # 角度编码（单位：0.1度 * 10）
        position = int(position_deg * 10)
        pos_high = (position >> 8) & 0xFF
        pos_low = position & 0xFF
        frame.append(pos_high)
        frame.append(pos_low)

        # 速度编码（单位：rad/s * 10）
        speed = int(speed_rad_per_s * 10)
        speed_high = (speed >> 8) & 0xFF
        speed_low = speed & 0xFF
        frame.append(speed_high)
        frame.append(speed_low)

        # BCC 校验：前10字节异或
        bcc = 0
        for byte in frame:
            bcc ^= byte
        frame.append(bcc)

        frame.append(0x7D)  # 帧尾

        return bytes(frame)
    
if __name__ == "__main__":
    gripper = New_Gripper("/dev/ttyACM1")
    # count = 1
    # while count < 10:
    #     time.sleep(8)
    #     direction = randrange(2)
    #     if direction:
    #         direction = "open"
    #     else:
    #         direction = "close"
    #     deg = randrange(1873)
    #     print(f"Desired direction: {direction}")
    #     print(f"Desired degree: {deg}")
    #     ps = time.time()
    #     gripper.move(direction, deg)
    #     pe = time.time()
    #     print(f"发送命令{count}次，耗时{pe-ps}秒")
    #     count += 1

    success = pyspacemouse.open(device="3Dconnexion Universal Receiver")
    if success:
        print("spacemouse connected")
    else:
        print("spacemouse connection failed")

    #gripper.send_command("open", 1872)

    # gripper.fully_open()

    while True:
        for _ in range(2000):
            state = pyspacemouse.read()
        # print(state)
        gripper.send_data(state)
        time.sleep(0.1)