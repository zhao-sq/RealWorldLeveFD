from crx_spacemouse_gripper import *
from cv_utils import findDevices, initialize_camera, get_rgbd_image_res
import keyboard
import time
import threading
import cv2
from pynput import keyboard as pynput_keyboard
import numpy as np
import os
import shutil
import tempfile

# Global state for pressed keys
_pressed_keys = set()

def _on_press(key):
    """Store pressed keys."""
    try:
        if hasattr(key, 'char') and key.char:  # e.g., 'q'
            _pressed_keys.add(key.char)
        elif key == pynput_keyboard.Key.enter:
            _pressed_keys.add('enter')
    except AttributeError:
        pass

def _on_release(key):
    """Remove released keys."""
    try:
        if hasattr(key, 'char') and key.char:
            _pressed_keys.discard(key.char)
        elif key == pynput_keyboard.Key.enter:
            _pressed_keys.discard('enter')
    except AttributeError:
        pass

# Start background listener (non-blocking, runs throughout the program)
_listener = pynput_keyboard.Listener(on_press=_on_press, on_release=_on_release)
_listener.start()

def wait_for_key(target_key):
    """Block until target_key is pressed once (like keyboard.wait)."""
    print(f"Press '{target_key}' to continue.")
    while True:
        if target_key in _pressed_keys:
            # Remove so it doesn't immediately trigger again
            _pressed_keys.discard(target_key)
            break
        time.sleep(0.01)

def is_key_pressed(target_key):
    """Check if a key is currently pressed (like keyboard.is_pressed)."""
    return target_key in _pressed_keys


class data_collector():
    def __init__(self, controller, cfg, gripper) -> None:
        self.data_path = None
        self.controller = controller
        self.cfg = cfg
        self.gripper = gripper

        self.in_hand_camera_pipeline = None
        self.in_hand_camera_align = None
        self.fixed_left_camera_pipeline = None
        self.fixed_left_camera_align = None
        self.fixed_right_camera_pipeline = None
        self.fixed_right_camera_align = None

        # init camera
        print("init cameras ...")
        self.init_camera()
        # self.ensenso = ensenso_cam()
        print("done")

    
    def init_camera(self):
        serials, ctx = findDevices()
        print(f"serials found: {serials}")
        self.in_hand_camera_pipeline, self.in_hand_camera_align = initialize_camera("337122070513", ctx)
        self.fixed_left_camera_pipeline, self.fixed_left_camera_align = initialize_camera("246322301968", ctx)
        self.fixed_right_camera_pipeline, self.fixed_right_camera_align = initialize_camera("241122306995", ctx)


    def collect_visual_data(self):
        threads = []
        # start the camera thread

        self.in_hand_camera_res = []
        threads.append(threading.Thread(target=get_rgbd_image_res, args=(self.in_hand_camera_pipeline, self.in_hand_camera_align, self.in_hand_camera_res)))
        self.fixed_left_camera_res = []
        threads.append(threading.Thread(target=get_rgbd_image_res, args=(self.fixed_left_camera_pipeline, self.fixed_left_camera_align, self.fixed_left_camera_res)))
        self.fixed_right_camera_res = []
        threads.append(threading.Thread(target=get_rgbd_image_res, args=(self.fixed_right_camera_pipeline, self.fixed_right_camera_align, self.fixed_right_camera_res)))


        for thread in threads:
            thread.start()
        print('Start camera threads.')

        for thread in threads:
            thread.join()
        print('Camera threads done.')


    def collect_data(self, time_step=1/10):
        # click the enter button to start the data collection
        # read the keyboard input
        print("Press 'Enter' to move to init position.")
        wait_for_key('enter')

        init_robot_pos(controller, TRIAL_INIT_JOINT_POS, 50)
        gripper.fully_open()

        print("Press 'Enter' to start data collection.")
        wait_for_key('enter')
        print("Data collection started. Press 'q' to stop.")

        count = 0
        current_state,_ = get_robot_state(controller)
        real_robot_state = current_state

        # record = False

        # press_button_time = time.time()
        try:
            while True:  # Start an infinite loop to collect data
                # Your data collection code goes here
                ps = time.time()
                current_state, real_robot_state, spacemouse_state = tel_ctl_spacemouse(controller, self.cfg, current_state, real_robot_state, SPEED, time_step, gripper, count=0)
                gripper_state = self.gripper.get_position_deg()
                is_gripper_in_action = gripper.is_in_action()
                print("robot state timestamp: ", time.time())
                print("robot state step time: ", time.time()-ps)

                self.collect_visual_data()
                print("step time: ", time.time()-ps)
                
                # print(is_gripper_in_action)
                
                # np.savez(self.data_path+str(count),sent_state=current_state,gripper_state=gripper_state,
                #          is_gripper_in_action=is_gripper_in_action,current_time=time.time())
                
                np.savez(self.data_path+str(count),sent_state=current_state,real_robot_state=real_robot_state,spacemouse_state=spacemouse_state,gripper_state=gripper_state,
                         is_gripper_in_action=is_gripper_in_action,current_time=time.time(),
                         in_hand_color=self.in_hand_camera_res[0],in_hand_depth=self.in_hand_camera_res[1],
                         fixed_left_color=self.fixed_left_camera_res[0],fixed_left_depth=self.fixed_left_camera_res[1],
                         fixed_right_color=self.fixed_right_camera_res[0],fixed_right_depth=self.fixed_right_camera_res[1])

                # np.savez(self.data_path+str(count),sent_state=current_state, real_robot_state=real_robot_state,gripper_state=gripper_state,
                #          is_gripper_in_action=is_gripper_in_action,current_time=time.time(),
                #          fixed_left_color=self.fixed_left_camera_res[0],fixed_left_depth=self.fixed_left_camera_res[1])

                # np.savez(self.data_path+str(count),sent_state=current_state, real_robot_state=real_robot_state,gripper_state=gripper_state,
                #          is_gripper_in_action=is_gripper_in_action,current_time=time.time(),
                #          in_hand_color=self.in_hand_camera_res[0],in_hand_depth=self.in_hand_camera_res[1],
                #          fixed_left_color=self.fixed_left_camera_res[0],fixed_left_depth=self.fixed_left_camera_res[1])

                def stack_frames(frames):
                    # Resize to same height for stacking
                    resized = [cv2.resize(f, (640, 480)) for f in frames if f is not None]
                    recolor = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in resized]
                    return cv2.hconcat(recolor) if recolor else None

                frames = [self.in_hand_camera_res[0], self.fixed_left_camera_res[0], self.fixed_right_camera_res[0]]
                # frames = [self.fixed_left_camera_res[0]]
                # frames = [self.in_hand_camera_res[0], self.fixed_left_camera_res[0]]
                stacked = stack_frames(frames)
                if stacked is not None:
                    cv2.imshow("All Cameras", stacked)
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        break

                count += 1
                if time_step-time.time()+ps > 0:
                    time.sleep(abs(time_step-time.time()+ps))
                else:
                    print('long time', time.time()-ps, 'expected time step', time_step)


                # if is_key_pressed('r'):  # Check if 'q' is pressed
                #     print("Record pos.")
                #     np.save(self.data_path + 'pos.npy', robot_pos)

                if is_key_pressed('q'):  # Check if 'q' is pressed
                    print("Stopping data collection.")
                    break

        except KeyboardInterrupt:
            print("Data collection stopped.")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    ENABLE_GRIPPER = True
    INIT_JOINT = np.array([0,0,0,0,-90,-30]) # Set the initial joint position of the robot.
    SPEED = 10 # Set the speed of the robot.
    TRIAL_INIT_JOINT_POS = np.array([0,0,0,0,-90,-30])

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
    # get the current state of robot
    current_state,_ = get_robot_state(controller)
    Cpos,cfg,_,_=controller.rmi.rmGetCartPos()
    
    dc = data_collector(controller, cfg, gripper)
    
    print('Warm up the cameras.')
    cost_time_list = []
    for i in range(50):
        init_time = time.time()
        dc.collect_visual_data()
        cost_time = time.time()-init_time
        print(i,"collecting time: ", cost_time)
        cost_time_list.append(cost_time)
        if len(cost_time_list) > 5:
            cost_time_list.pop(0)
        if i > 10 and np.std(cost_time_list) < 0.2:
            print(cost_time_list)
            break
    print('Warm up is done.')
    
    folder = "collected_data/trash_disposal/"
    exp_name = "demo_"
    num_demos = 3

    # ensure parent folder exists
    os.makedirs(folder, exist_ok=True)
    success_dir = os.path.join(folder, "success")
    failure_dir = os.path.join(folder, "failure")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    def next_demo_path(label):
        label_dir = os.path.join(folder, label)
        existing_indices = set()
        for d in os.listdir(label_dir):
            if not d.startswith(exp_name):
                continue
            try:
                existing_indices.add(int(d[len(exp_name):]))
            except ValueError:
                continue

        idx = 0
        while idx in existing_indices:
            idx += 1
        return os.path.join(label_dir, f"{exp_name}{idx}")

    # helper that asks the user to label the last demo and move it accordingly
    def classify_demo(path):
        print("Press 's' for success, 'f' for failure for demo:", path)
        # block until either key is pressed
        while True:
            if is_key_pressed('s'):
                label = 'success'
                _pressed_keys.discard('s')
                break
            if is_key_pressed('f'):
                label = 'failure'
                _pressed_keys.discard('f')
                break
            time.sleep(0.01)

        new_path = next_demo_path(label)
        # move the demo folder into the appropriate subfolder
        try:
            shutil.move(path, new_path)
        except Exception as e:
            print(f"Failed to move {path} -> {new_path}: {e}")
            new_path = path
        with open(os.path.join(new_path, 'label.txt'), 'w') as f:
            f.write(label)
        print(f"Demo {path} classified as {label}, moved to {new_path}.")
        return label

    for entry in os.listdir(folder):
        entry_path = os.path.join(folder, entry)
        if entry.startswith("recording_") and os.path.isdir(entry_path):
            shutil.rmtree(entry_path, ignore_errors=True)

    collected = 0
    # loop until we've attempted num_demos new recordings
    while collected < num_demos:
        demo_path = tempfile.mkdtemp(prefix="recording_", dir=folder)
        dc.data_path = demo_path + "/"
        try:
            dc.collect_data()
            print(dc.data_path, "done.")
            # classification step (returns label in case you need it)
            label = classify_demo(dc.data_path)
            if label == 'success':
                collected += 1
        finally:
            if os.path.isdir(demo_path):
                shutil.rmtree(demo_path, ignore_errors=True)

    controller.close()
    pread.terminate()
    
