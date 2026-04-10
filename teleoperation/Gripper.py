import time

class Gripper():
    def __init__(self, ser, gripper_operation_time):
        """Initialize gripper.

        Args:
            ser: Serial port.
            gripper_operation_time (time): Time for gripper to open or close.
        """
        self.ser = ser
        self.operation_time = gripper_operation_time
        self.state = 0
        self.action = None
        self.data_last_sent_time = time.time() - self.operation_time

    def open(self):
        if self.state != 0:
            self.action = '0'
            self.get_ser().write(self.get_action().encode())
            self.data_last_sent_time = time.time()
            self.action = None
            self.state = 0

    def close(self):
        if self.state != 1:
            self.action = '1'
            self.get_ser().write(self.get_action().encode())
            self.data_last_sent_time = time.time()
            self.action = None
            self.state = 1

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
        
    def get_state(self):
        """
        Returns:
            state (int): The current state of the gripper. 1 means close and 0 means open. 
        """
        return self.state
    
    def get_action(self):
        """
        Returns:
            self.action (int): The action we want the gripper to execute. 1 means close, 0 means open, None means do nothing.
        """
        return self.action
    
    def get_data_last_sent_time(self):
        """
        Returns:
            self.data_last_sent_time (time): The time we last operate the gripper.
        """
        return self.data_last_sent_time
    
    def activate(self, spacemouse_state):
        """Set self.action to 0 or 1 if the right button of the spacemouse is pressed.

        Args:
            spacemouse_state: The current state of the spacemouse.
        """ 
        if spacemouse_state.buttons[-1]:
            print("gripper")
            if self.get_state() == 0:
                self.action = '1'
            elif self.get_state() == 1:
                self.action = '0'

    def send_data(self, spacemouse_state):
        """Send data to the gripper and update self.state, self.action, self.data_last_sent_time if self.action is not None.
        """
        self.activate(spacemouse_state)
        if self.get_action():
            self.get_ser().write(self.get_action().encode())
            if self.get_state() == 0:
                self.state = 1
            elif self.get_state() == 1:
                self.state = 0
            self.action = None
            self.data_last_sent_time = time.time()

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