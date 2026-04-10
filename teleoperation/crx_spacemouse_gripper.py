import numpy as np
from new_gripper import New_Gripper
from crx_controller import crx_controller
from crx_utils import pyRMI, pyHSPOServer
from crx_utils.teleop_data_collection import keymap_xbox as keymapfcn
import matplotlib.pyplot as plt
import time
import pygame
from scipy.spatial.transform import Rotation
from crx_utils import crxTrqUtil
from crx_utils import readJntTrq, readJntPos, moveJnt
import zmq
import sys
import multiprocessing
import copy
import os
import math
import pyspacemouse
from collections import namedtuple
import termios
import tty
import threading
import queue

import numpy as np

import serial
import fcntl

ENABLE_GRIPPER = True # Set to True if we want to operate the gripper.
INIT_JOINT = np.array([0,0,0,0,-90,-30]) # Set the initial joint position of the robot.
SPEED = 10 # Set the speed of the robot.
TEST_STATE_STEPS_PER_DIRECTION = 10
TEST_SPACEMOUSE_STATES = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, -1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, -1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, -1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)



# Shared spacemouse state
last_spacemouse_state = None
last_spacemouse_time = 0.0
state_lock = threading.Lock()

# Shared robot state
last_robot_state = None
last_robot_response = None
last_robot_time = 0.0
robot_state_lock = threading.Lock()

def spacemouse_callback(state):
    global last_spacemouse_state, last_spacemouse_time
    with state_lock:
        last_spacemouse_state = state
        last_spacemouse_time = time.time()
    time.sleep(0.001)

def robot_state_callback(controller):
    """Callback function to continuously update robot state in a separate thread."""
    global last_robot_state, last_robot_response, last_robot_time
    while True:
        try:
            robot_state, response = get_robot_state(controller)
            with robot_state_lock:
                last_robot_state = robot_state
                last_robot_response = response
                last_robot_time = time.time()
            time.sleep(1/100)  # 100Hz update rate
        except Exception as e:
            print(f"Error reading robot state: {e}")
            time.sleep(0.1)  # Wait longer on error

def connect_to_robot(FTdata, init_joint):
    """Connect to the robot and initialize the position of the robot.
    
    Args:
        FTdata

    Returns:
        crx_controller: The controller of the robot.
    """
    controller = crx_controller(FTdata)
    print("robot connected")
    init_robot_pos(controller, init_joint, 50)
    return controller

def init_robot_pos(controller, init_joint, speed):
    """Initialize the position of the robot.

    Args:
        controller (crx_controller): The controller of the robot.
        init_joint (np.array): The initial position.
    """
    controller.move_to_joint(init_joint, speed)
    
def get_robot_state(controller):
    """Get the current state of the robot.

    Args:
        controller (crx_controller): The controller of the robot.

    Returns:
        np.array: The current state of the robot.
    """
    return controller.get_robot_state()

def connect_to_spacemouse(device_name="3Dconnexion Universal Receiver"):
    """Connect to the spacemouse.

    Args:
        device_name (str, optional): The name of the spacemouse. Defaults to "3Dconnexion Universal Receiver".

    Returns:
        boolean: True if we successfully connected to spacemouse. False if connection failed.
    """
    success = pyspacemouse.open(device=device_name, callback=spacemouse_callback)
    if success:
        print("spacemouse connected")
    else:
        print("spacemouse connection failed")
        return success

    def polling():
        while True:
            try:
                pyspacemouse.read()
            except Exception:
                break

    t = threading.Thread(target=polling, daemon=True)
    t.start()
    global last_spacemouse_state, last_spacemouse_time
    last_spacemouse_state = pyspacemouse.read()
    last_spacemouse_time = time.time()
    return success

def connect_to_robot_state_thread(controller):
    """Start a thread to continuously read robot state.
    
    Args:
        controller (crx_controller): The controller of the robot.
        
    Returns:
        boolean: True if thread started successfully.
    """
    try:
        # Initialize robot state
        global last_robot_state, last_robot_response, last_robot_time
        robot_state, response = get_robot_state(controller)
        with robot_state_lock:
            last_robot_state = robot_state
            last_robot_response = response
            last_robot_time = time.time()
        
        # Start the robot state reading thread
        robot_thread = threading.Thread(target=robot_state_callback, args=(controller,), daemon=True)
        robot_thread.start()
        print("robot state thread started")
        return True
    except Exception as e:
        print(f"Failed to start robot state thread: {e}")
        return False

def read_spacemouse_state():
    """Get the current state of the spacemouse.

    Returns:
        The current state of the spacemouse.
    """
    return last_spacemouse_state

    # states = []

    # for _ in range(100):
    #     state = pyspacemouse.read()
    #     states.append([
    #         state.x, state.y, state.z,
    #         state.roll, state.pitch, state.yaw,
    #         state.buttons[0], state.buttons[1]
    #     ])

    # states = np.array(states)
    # agg = states.mean(axis=0)

    # State = namedtuple("State", ["x", "y", "z", "roll", "pitch", "yaw", "buttons"])
    # return State(
    #     x=agg[0],
    #     y=agg[1],
    #     z=agg[2],
    #     roll=agg[3],
    #     pitch=agg[4],
    #     yaw=agg[5],
    #     buttons=[int(agg[6] > 0), int(agg[7] > 0)]
    # )

def read_robot_state():
    """Get the current state of the robot from the continuously updated thread.

    Returns:
        tuple: (robot_state, response, timestamp) - The current robot state, response dict, and timestamp.
               Returns (None, None, None) if no state is available.
    """
    with robot_state_lock:
        return last_robot_state, last_robot_response, last_robot_time
    
def prepare_exit(state, prepare_exit_time):
    if state.buttons[0] and state.buttons[-1]:
        if prepare_exit_time is None:
            return time.time()
        else:
            return prepare_exit_time
    return None
    
def exit(prepare_exit_time):
    if prepare_exit_time:
        current_time = time.time()
        time_passed = current_time - prepare_exit_time
        if time_passed >= 3:
            return True
    return False
        
def get_desired_robot_state(state, current_state, speed):
    """Calculate the desired robot state.

    Args:
        state: The current state of the spacemouse.
        mode (list): It records the orientation mode and gripper mode.
        current_state (np.array): The current state of the robot.
        speed (int): Speed of the robot.

    Returns:
        desired_state(np.array): The desired state of the robot.
    """
    if state.buttons[0]:
        vel_mouse = np.array([state.x, state.y, state.z, -state.pitch, state.roll, -state.yaw])
        pos_offset = 0.01
        rotation_offset = np.array([5, 5, 5]) * np.pi / 180
    else:
        # since the spacemouse is too sensitive in the -z direction, we will only be reading the z
        # value of the spacemouse after it reach a certain value
        z = 0
        if state.z == -1.0 or state.z >= 0 or True:
            z = state.z
        yaw = 0
        if abs(state.yaw) >= 1.0:
            yaw = state.yaw
        vel_mouse = np.array([state.x, state.y, z, -state.pitch, state.roll, -yaw])

        pos_offset = speed
        rotation_offset = np.array([0, 0, 7]) * np.pi /180
    
    # if current_state[2] > 65:
    #     vel_mouse[2] = -1
    # else:
    #     vel_mouse[2] = 0

    delta_pos = vel_mouse[:3] * pos_offset
    # transfer rol pitch yaw to rotation matrix
    delta_ori = Rotation.from_euler("xyz", np.multiply(vel_mouse[3:], rotation_offset)).as_matrix()
    
    # transfer roll pitch yaw to rotation matrix
    robot_ori = Rotation.from_euler("xyz",np.array([current_state[3],current_state[4],current_state[5]])/180*np.pi).as_matrix()
    desired_pos = current_state[:3] + delta_pos
    desired_ori = delta_ori @ robot_ori
    # transfer desired_ori to xyz euler angle
    desired_ori = Rotation.from_matrix(desired_ori).as_euler("xyz")
    
    desired_state = np.concatenate([desired_pos, desired_ori / np.pi * 180], axis=0)
    
    return desired_state

def getJntTrq(hspo):
    jtrq=None
    cnt=0
    while jtrq is None and cnt<10:
        packet = hspo.sock.recvfrom(1024)[0]
        data = hspo.processPacket(packet)
        if "variable" in data:
            jtrq=data["variable"]
        cnt+=1
    if jtrq is None:
        raise SystemError("Reading torque time out!")
    return jtrq

def getJntPosTrq(hspo):
    jtrq=None
    jpos=None
    cnt=0
    while ((jtrq is None) or (jpos is None)) and cnt<10:
        packet = hspo.sock.recvfrom(1024)[0]
        data = hspo.processPacket(packet)
        if "position" in data:
            jpos=data["position"]
        if "variable" in data:
            jtrq=data["variable"]
        cnt+=1
    if jtrq is None or jpos is None:
        raise SystemError('Reading jnt position and torque time out!')
    return jpos, jtrq

def getExtForce(hspo,trqutil):
    jpos,jtrq=getJntPosTrq(hspo)
    F_ext_ee=trqutil.extFT_simple(jtrq,jpos) # use _simple in experiment, ee for end effector, no ee for world frame
    F_ext_ee *= -1 # adjust sensor reading direction
    return F_ext_ee

def hspoRead(FTdata,trqutil):

    " initialize and connect hspo "
    hspo = pyHSPOServer() # hspo
    hspo.hspoConnect()

    jtrq=getJntTrq(hspo)
    trqoffset=trqutil.calTrqOffset_simple(jtrq)
    trqutil.settrqoffset(trqoffset)
    print('='*100)
    print(' ')
    print(f'Jnt torque offset: {trqoffset}')
    print(' ')
    print('='*100)

    while True:
        F_ext_ee=getExtForce(hspo,trqutil)
        with FTdata.get_lock():
            for i in range(6):
                FTdata[i]=F_ext_ee[i]

def connect_to_gripper(enable_gripper, serial_port):
    """Connect to the gripper.
    
    Args:
        enable_griipper (boolean): True if we want to operate gripper and False otherwise.

    Returns:
        gripper (Gripper)
    """
    if enable_gripper:
        gripper = New_Gripper(serial_port)
    else:
        gripper = None
    return gripper

def spacemouse_state_to_robot_state(controller, cfg, current_state, real_robot_state, speed, dt, gripper=None, count=None):
    """Transfer spacemouse state to the desired robot state.

    Args:
        controller (crx_controller): The cotroller of the robot.
        current_state (np.array): The current state of the robot.
        speed (int): Speed of the robot.
        gripper (Gripper, optional): The gripper we want to operate. Defaults to None.
    Returns:
        desired_state (np.array): The desired state of the robot.
    """
    # state_index = min(count // TEST_STATE_STEPS_PER_DIRECTION, len(TEST_SPACEMOUSE_STATES) - 1)
    # spacemouse_state = TEST_SPACEMOUSE_STATES[state_index]
    # State = namedtuple("State", ["x", "y", "z", "pitch", "roll", "yaw", "buttons"])
    # state = State(
    #     x=spacemouse_state[0],
    #     y=spacemouse_state[1],
    #     z=spacemouse_state[2],
    #     pitch=spacemouse_state[3],
    #     roll=spacemouse_state[4],
    #     yaw=spacemouse_state[5],
    #     buttons=[int(spacemouse_state[6]), int(spacemouse_state[7])],
    # )
    state = read_spacemouse_state()
    spacemouse_state = np.array([state.x, state.y, state.z, state.pitch, state.roll, state.yaw, state.buttons[0], state.buttons[-1]])

    # cnt = 0
    # while cnt<1000 and controller.rmi.isLastLastSentMotionNotDone():
    #     # rmi.rmGetStatus()
    #     # Cpos,cfg,robtime=self.rmi.rmGetCartPos()
    #     get_robot_start_time = time.time()
    #     real_robot_state, response = get_robot_state(controller)
    #     # print('*' * 50)
    #     # print(read_robot_state())
    #     # print('*' * 50)
    #     get_robot_end_time = time.time()
    #     # states.append(response["Position"]["Y"])
    #     # times.append(response["TimeTag"])
    #     cnt+=1
    #     print('get robot state time', get_robot_end_time - get_robot_start_time)
    #     print('real robot state', real_robot_state[:3])
    #     break

    real_robot_state, response = get_robot_state(controller)

    if state.buttons[-1]:
        gripper.send_data(state)
        desired_state = current_state
    else:
        # switch to oriantation mode or translation mode if the left button on the spacemouse is pressed
        # get the desired state of the robot
        print('space mouse', state.x,state.y,state.z)    
        
        # desired_state = get_desired_robot_state(state, real_robot_state, speed)
        desired_state = get_desired_robot_state(state, current_state, speed)
        # if count == 10:
        #     desired_state[1] += 10
        # elif count == 11:
        #     desired_state[2] += 10
        
        # send data to gripper if the right button on the spacemouse is pressed
            
        # norm = np.sum([(desired_state[i] - current_state[i])**2 for i in range(3)]) ** 0.5
        
        print('-'*20)
        # print('norm',norm)
        # print(dt * 100)

        # spd = norm / dt

        # Debuging
        # print('#'*50)
        # print("Debuging")
        # print("current robot state:", current_state)
        # print("current mouse state:", state)
        # print("gripper state:", gripper.get_state())

    print('desired state', desired_state[:3])
    controller.rmi.rmLinearMotion(desired_state,cfg,termType='CNT',termVal=100,spdType="mmSec",spd=900)
    
    return desired_state, real_robot_state, spacemouse_state

def set_robot_state(controller, cfg, set_state, dt):
    controller.rmi.rmLinearMotion(set_state,cfg,termType='CNT',termVal=100,spdType="mmSec",spd=900)
    

def tel_ctl_spacemouse(controller, cfg, current_state, real_robot_state, SPEED, dt, gripper, count):
    t_1 = time.time()
    current_state, real_robot_state, spacemouse_state = spacemouse_state_to_robot_state(controller, cfg, current_state, real_robot_state, SPEED, dt, gripper, count)
    t_2 = time.time()
    cnt=0
    while cnt<1000 and controller.rmi.isLastLastSentMotionNotDone():
        # rmi.rmGetStatus()
        controller.rmi.rmRecievePacket()
        # Cpos,cfg,robtime=self.rmi.rmGetCartPos()
        cnt+=1
        
    print("-"*20)
    print(cnt, not controller.rmi.isLastLastSentMotionNotDone())

    t_3 = time.time()
    # print("compute pos time: ", t_2 - t_1)
    # print("send time: ", t_3 - t_2)

    return current_state, real_robot_state, spacemouse_state


if __name__ == "__main__":
    trqutil=crxTrqUtil() # trqutil
    # update trq util model
    trqutil.tool['position'] = np.array([0.,0,0*138.6/1000])
    trqutil.tool['orientation'] = Rotation.from_euler('ZYX',[0,0,-180],degrees=True).as_quat()
    # additional setting for exp
    trqutil.LinkDynParam['Mass']=np.zeros(6)
    trqutil.Ml = np.array(trqutil.LinkDynParam['Mass'])
    FTdata=multiprocessing.Array('f',np.zeros(6))
    " define threads for reading trq data and teleoperation "
    pread=multiprocessing.Process(target=hspoRead,args=(FTdata,trqutil,))
    pread.start()
    
    " connect to robot "
    # train koopman operator:
    
    # init controller
    controller = connect_to_robot(FTdata, INIT_JOINT)
    # connect_to_robot_state_thread(controller)
    # connect to spacemouse
    success = connect_to_spacemouse()
    # connect to gripper
    gripper = connect_to_gripper(ENABLE_GRIPPER, serial_port="/dev/ttyACM1")
    # get the current state of robot
    current_state, _ = get_robot_state(controller)
    start_real_robot_state, _ = get_robot_state(controller)
    print("start real robot state:", start_real_robot_state)
    real_robot_state = current_state
    Cpos,cfg,_,_=controller.rmi.rmGetCartPos()
    
    prepare_exit_time = None
    dt = 1/10

    states = []
    times = []  # store timestamps
    save_dir = "crx_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    count = 0
    while success:
        # if count >= len(TEST_SPACEMOUSE_STATES) * TEST_STATE_STEPS_PER_DIRECTION:
        #     print("Finished executing all TEST_SPACEMOUSE_STATES commands.")
        #     break

        # print('*'*20 , count, '*'*20)

        ps = time.time()
        

        current_state, real_robot_state, spacemouse_state = tel_ctl_spacemouse(controller, cfg, current_state, real_robot_state, SPEED, dt, gripper, count=count)
        # print("current_state: ")
        # print(current_state)
            
        cost_time = time.time() - ps
                
        if cost_time > dt-0.0001:
            print('cost time:', cost_time, 'expected frequency:', int(1/dt))
        else:
            time.sleep(dt - cost_time)
            
        prepare_exit_time = prepare_exit(read_spacemouse_state(), prepare_exit_time)
        exit_session = exit(prepare_exit_time)
        if exit_session:
            break
        count += 1
        # # if count > 2:
        # #     SPEED = 0

        # if count > 15:
        #     break

    end_real_robot_state = None
    # for _ in range(5):
    #     time.sleep(0.1)
    #     end_real_robot_state, _ = get_robot_state(controller)
    # print("start real robot state:", start_real_robot_state)
    # print("end real robot state:", end_real_robot_state)

    states = np.array(states)
    times = np.array(times) * 0.00025
    labels = ["y"]

    plt.figure(figsize=(10, 6))
    plt.plot(times, states, label=labels)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Crx State Over Time")
    plt.legend()
    plt.grid(True)

    # save_path = os.path.join(save_dir, "fine_state_time_plot_pitch.png")
    # plt.savefig(save_path)
    plt.close()

    # --- Compute derivative dy/dt ---
    velocities = np.gradient(states, times)  # Element-wise dy/dt

    # --- Plot v(t) ---
    plt.figure(figsize=(10, 6))
    plt.plot(times, velocities, label="dy/dt", color='orange')
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.title("Crx State Velocity Over Time")
    plt.legend()
    plt.grid(True)

    # save_path = os.path.join(save_dir, "fine_state_velocity_plot_pitch.png")
    # plt.savefig(save_path)
    plt.close()
                
    # --- Compute acceleration d²y/dt² ---
    accelerations = np.gradient(velocities, times)

    # --- Plot a(t) ---
    plt.figure(figsize=(10, 6))
    plt.plot(times, accelerations, label="d²y/dt²", color='green')
    plt.xlabel("Tag")
    plt.ylabel("Acceleration")
    plt.title("Crx State Acceleration Over Time")
    plt.legend()
    plt.grid(True)

    # save_path = os.path.join(save_dir, "fine_state_acceleration_plot_pitch.png")
    # plt.savefig(save_path)
    plt.close()

    controller.close()
    pread.terminate()
    
