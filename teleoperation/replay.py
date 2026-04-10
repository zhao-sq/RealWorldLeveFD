from crx_spacemouse_gripper import *
import keyboard
import time
import threading
import numpy as np
import os

class spacemouse():
    def __init__(self, state):
        self.x = state[0]
        self.y = state[1]
        self.z = state[2]
        self.pitch = state[3]
        self.roll = state[4]
        self.yaw = state[5]
        self.buttons = [state[6], state[7]]


if __name__ == "__main__":
    ENABLE_GRIPPER = True
    INIT_JOINT =  np. array([15,40,-25,0,-70,-60]) # np.array([10,30,-20,0,-70,60]) # Set the initial joint position of the robot.
    SPEED = 30 # Set the speed of the robot.
    dt = 1/10

    # Shared spacemouse state
    last_spacemouse_state = None
    last_spacemouse_time = 0.0
    state_lock = threading.Lock()
    
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
    # connect to spacemouse
    success = connect_to_spacemouse()
    if not success:
        raise Exception("spacemouse not connected")
    # connect to gripper
    gripper = connect_to_gripper(ENABLE_GRIPPER, serial_port="/dev/ttyACM1")

    controller.move_to_joint(INIT_JOINT, 50)
    
    # get the current state of robot
    cnt = 0
    current_state,_ = get_robot_state(controller)
    Cpos,cfg,robtime,_ = controller.rmi.rmGetCartPos()
    while cnt<1000 and controller.rmi.isLastLastSentMotionNotDone():
        Cpos,cfg,robtime,_ = controller.rmi.rmGetCartPos()
        current_state, response = get_robot_state(controller)
        cnt+=1

    
    replay_directory = "/home/msc/Documents/crx_rmi_utils/teleoperation/collected_data/simp_pour_water/demo_0"
    data_count = len([name for name in os.listdir(replay_directory) if os.path.isfile(os.path.join(replay_directory, name))])

    time.sleep(3)
    
    for i in range(data_count):
        ps = time.time()
        data = np.load(replay_directory + "/" + str(i) + ".npz")
        # desired_state = data['sent_state']
        # print('#' * 50)
        # print('sent_state:', desired_state)
        spacemouse_state = data['spacemouse_state']
        print("space mouse:", spacemouse_state)
        spacemouse_state = spacemouse(spacemouse_state)
        # print('spacmouse2robotstate:', get_desired_robot_state(spacemouse_state, current_state, SPEED))
        if spacemouse_state.buttons[-1]:
            gripper.send_data(spacemouse_state)
            desired_state = current_state
        else:
            desired_state = get_desired_robot_state(spacemouse_state, current_state, SPEED)
        current_state = desired_state
        controller.rmi.rmLinearMotion(desired_state,cfg,termType='CNT',termVal=100,spdType="mmSec",spd=900)
        real_robot_state, response = get_robot_state(controller)
        print('real robot state', real_robot_state[:3])

        cnt=0
        while cnt<1000 and controller.rmi.isLastLastSentMotionNotDone():
            controller.rmi.rmRecievePacket()
            cnt+=1
                 
        cost_time = time.time() - ps
        if cost_time > dt:
            print('cost time:', cost_time, 'expected frequency:', int(1/dt))
        else:
            time.sleep(dt - cost_time)
                
    controller.close()
    pread.terminate()
    