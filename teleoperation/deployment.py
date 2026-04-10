#!/usr/bin/env python3
import time
import threading
import multiprocessing
import numpy as np
from pathlib import Path

import cv2
from scipy.spatial.transform import Rotation as R
import torch

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# --- your robot / IO files ---
from crx_controller import crx_controller
from crx_spacemouse_gripper import *
from new_gripper import New_Gripper
from cv_utils import findDevices, initialize_camera, get_rgbd_image_res

from pynput import keyboard as pynput_keyboard
from crx_utils import *

# =========================
# Config
# =========================
GRIPPER_SERIAL = "/dev/ttyACM1"

INIT_JOINT = np.array([0,0,0,0,-90,-30])
RESET_SPEED = 50
SPEED = 15 # 30
DT = 1 / 10

PRETRAINED_POLICY = Path("/home/msc/Documents/davidwu/robomimic-lerobot-pipeline/lerobot/lerobot/scripts/outputs/diffusion_real_crx_mid_leftrightcam_pour_water/checkpoints/last/pretrained_model")
# LEFT_PRETRAINED_POLICY = Path("/home/msc/Documents/davidwu/robomimic-lerobot-pipeline/lerobot/lerobot/scripts/outputs/diffusion_real_crx_mid_leftcam_pour_water/checkpoints/last/pretrained_model")
# RIGHT_PRETRAINED_POLICY = Path("/home/msc/Documents/davidwu/robomimic-lerobot-pipeline/lerobot/lerobot/scripts/outputs/diffusion_real_crx_mid_rightcam_pour_water/checkpoints/last/pretrained_model")
NUM_STEPS = 200

# Camera serials (update if yours differ)
SER_IN_HAND  = "337122070513"
SER_LEFT     = "246322301968"
SER_RIGHT    = "241122306995"

IMG_H = 224
IMG_W = 224

GRIPPER_DEG_MIN = 0
GRIPPER_DEG_MAX = 1872

TRIAL_PATH = "/home/msc/Documents/crx_rmi_utils/teleoperation/collected_data/mid_pour_water/demo_0"

spacemouse_buffer = []

# =========================
# Key watcher
# =========================
class KeyWatcher:
    def __init__(self):
        self._pressed = set()
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        try:
            if hasattr(key, "char") and key.char:
                self._pressed.add(key.char.lower())
        except Exception:
            pass
        if key == pynput_keyboard.Key.enter:
            self._pressed.add("enter")

    def _on_release(self, key):
        try:
            if hasattr(key, "char") and key.char:
                self._pressed.discard(key.char.lower())
        except Exception:
            pass
        if key == pynput_keyboard.Key.enter:
            self._pressed.discard("enter")

    def is_pressed(self, k: str) -> bool:
        return k in self._pressed

    def wait_for(self, k: str):
        while not self.is_pressed(k):
            time.sleep(0.01)
        while self.is_pressed(k):
            time.sleep(0.01)

class spacemouse():
    def __init__(self, state):
        self.x = state[0]
        self.y = state[1]
        self.z = state[2]
        self.pitch = state[3]
        self.roll = state[4]
        self.yaw = state[5]
        button_1 = 1.0 if state[6] >= 0.5 else 0.0
        button_2 = 1.0 if state[7] >= 0.5 else 0.0
        # if button_2 == 1.0:
        #     spacemouse_buffer.append(button_2)
        # if len(spacemouse_buffer) > 20:
        #     button_2 = 0.0
        self.buttons = [button_1, button_2]
        # self.buttons = [state[6], state[7]]

# =========================
# Pose math (world-frame deltas)
# =========================
def compose_delta_next_pose_world(current_pose_deg6, delta):
    """
    World-frame composition (positions in mm, angles in deg):
      p_{t+1} = p_t + Δp_world
      R_{t+1} = R(ΔEuler) @ R_t
    """
    x, y, z, Wdeg, Pdeg, Rdeg = current_pose_deg6
    dp_mm   = np.array(delta[:3], dtype=float)
    dth_deg = np.array(delta[3:6], dtype=float)

    R_t = R.from_euler("xyz", np.deg2rad([Wdeg, Pdeg, Rdeg])).as_matrix()
    R_d = R.from_euler("xyz", np.deg2rad(dth_deg)).as_matrix()

    p_next = np.array([x, y, z], dtype=float) + dp_mm
    R_next = R_d @ R_t

    WPR_next_rad = R.from_matrix(R_next).as_euler("xyz")
    WPR_next_deg = np.rad2deg(WPR_next_rad)
    return np.concatenate([p_next, WPR_next_deg]).astype(np.float64)

# =========================
# Camera helpers
# =========================
def _resize_image(img_hwc):
    """
    Expect HxWx3 (uint8). Resize to 224x224, convert to CHW float32 [0,1] format.
    This mimics the transforms.ToTensor() behavior from LeRobot pipeline.
    """
    if img_hwc is None:
        return None
    if img_hwc.ndim != 3 or img_hwc.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got {None if img_hwc is None else img_hwc.shape}")
    if (img_hwc.shape[0], img_hwc.shape[1]) != (IMG_H, IMG_W):
        img_hwc = cv2.resize(img_hwc, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    
    # HWC -> CHW
    img_chw = np.transpose(img_hwc, (2, 0, 1)).copy()
    
    # Convert uint8 [0,255] -> float32 [0,1] (mimics transforms.ToTensor())
    img_chw = img_chw.astype(np.float32) / 255.0
    
    return img_chw

# =========================
# Env: state(float32) + 3× images (HxWx3 uint8)
# =========================
class RealEnv:
    def __init__(self, controller: crx_controller, gripper: New_Gripper, device="cuda"):
        self.controller = controller
        self.gripper = gripper
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Cameras
        serials, ctx = findDevices()
        print(f"serials found: {serials}")
        self.in_hand_pipe,  self.in_hand_align  = initialize_camera(SER_IN_HAND, ctx)
        self.left_pipe,     self.left_align     = initialize_camera(SER_LEFT, ctx)
        self.right_pipe,    self.right_align    = initialize_camera(SER_RIGHT, ctx)

    def _collect_visual_data(self):
        self._in_hand_res  = []
        self._left_res     = []
        self._right_res    = []
        threads = [
            threading.Thread(target=get_rgbd_image_res, args=(self.in_hand_pipe, self.in_hand_align, self._in_hand_res)),
            threading.Thread(target=get_rgbd_image_res, args=(self.left_pipe,    self.left_align,    self._left_res)),
            threading.Thread(target=get_rgbd_image_res, args=(self.right_pipe,   self.right_align,   self._right_res)),
        ]
        for th in threads: th.start()
        for th in threads: th.join()

    def get_obs(self, time_step, to_device=True, from_dataset=False):
        # --- state (float32) ---
        # cnt=0
        # while cnt<1000 and self.controller.rmi.isLastLastSentMotionNotDone():
        #     pose, _ = self.controller.get_robot_state()         # [X,Y,Z,W,P,R] mm/deg
        #     cnt += 1

        desired_action = None
        if from_dataset:
            data = np.load(TRIAL_PATH + "/" + str(time_step) + ".npz")
            pose = data["real_robot_state"] * 0.0
            pos_deg = data["gripper_state"] * 0.0
            print("Gripper:", pos_deg)
            in_hand_rgb = data["in_hand_color"]
            left_rgb = data["fixed_left_color"]
            right_rgb = data["fixed_right_color"]
            desired_action = data['spacemouse_state']
            print("desired action:", desired_action)
        else:
            # pose, _ = self.controller.get_robot_state()         # [X,Y,Z,W,P,R] mm/deg
            # pos_deg = self.gripper.get_position_deg()
            pose = [0.0] * 6
            pos_deg = 0.0
            self._collect_visual_data()
            in_hand_rgb = self._in_hand_res[0]
            left_rgb = self._left_res[0]
            right_rgb = self._right_res[0]

        def stack_frames(frames):
            # Resize to same height for stacking
            resized = [cv2.resize(f, (424, 240)) for f in frames if f is not None]
            return cv2.hconcat(resized) if resized else None

        frames = [in_hand_rgb, left_rgb, right_rgb]
        print("cameras", [f.shape for f in frames])
        stacked = stack_frames(frames)
        if stacked is not None:
            cv2.imshow("All Cameras", stacked)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                ...

        pose = np.asarray(pose, dtype=np.float32)
        pos_deg = float(pos_deg)

        state_vec = np.concatenate([pose, [pos_deg]], dtype=np.float32).reshape((7,))
        state_out = torch.from_numpy(state_vec).to(self.device, dtype=torch.float32).unsqueeze(0) if to_device else state_vec

        # --- images (HxWx3 uint8) ---
        in_hand_rgb = _resize_image(in_hand_rgb)   # HxWx3 uint8
        left_rgb    = _resize_image(left_rgb)
        right_rgb   = _resize_image(right_rgb)

        if to_device:
            # convert to torch **without** changing layout or dtype
            in_hand_rgb = torch.from_numpy(in_hand_rgb).to(self.device, dtype=torch.float32).unsqueeze(0)
            left_rgb    = torch.from_numpy(left_rgb).to(self.device, dtype=torch.float32).unsqueeze(0)
            right_rgb   = torch.from_numpy(right_rgb).to(self.device, dtype=torch.float32).unsqueeze(0)

        # fixed_rgb = right_rgb


        # Build obs dict. Images stay HxWx3 uint8.
        # return {
        #     "observation.state": torch.cat([state_out, state_out], dim=0),             # float32 (torch if to_device)
        #     # "observation.images.in_hand": in_hand_rgb,  # 3xHxW float32
        #     # "observation.images.fixed_left": left_rgb,  # 3xHxW float32
        #     # "observation.images.fixed_right": right_rgb # 3xHxW float32
        #     "observation.images.fixed_left_right": torch.cat([left_rgb, right_rgb], dim=0),  # 3xHxW float32
        # }, desired_action

        return {
            "observation.state": state_out,             # float32 (torch if to_device)
            "observation.images.in_hand": in_hand_rgb,  # 3xHxW float32
            "observation.images.fixed_left": left_rgb,  # 3xHxW float32
            # "observation.images.fixed_right": right_rgb # 3xHxW float32
            # "observation.images.fixed_left_right": fixed_rgb,  # 3xHxW float32
        }, desired_action

        # return {
        #     "observation.state": state_out,             # float32 (torch if to_device)
        #     # "observation.images.in_hand": in_hand_rgb,  # 3xHxW float32
        #     "observation.images.fixed_left": left_rgb,  # 3xHxW float32
        #     # "observation.images.fixed_right": right_rgb # 3xHxW float32
        #     # "observation.images.fixed_left_right": fixed_rgb,  # 3xHxW float32
        # }, {
        #     "observation.state": state_out,             # float32 (torch if to_device)
        #     # "observation.images.in_hand": in_hand_rgb,  # 3xHxW float32
        #     # "observation.images.fixed_left": left_rgb,  # 3xHxW float32
        #     "observation.images.fixed_right": right_rgb # 3xHxW float32
        #     # "observation.images.fixed_left_right": fixed_rgb,  # 3xHxW float32
        # }, desired_action

    def exec_zero_actions(self, cfg, dt=DT):
        for _ in range(10):
            t0 = time.time()
            pose, _ = self.controller.get_robot_state()
            self.controller.rmi.rmLinearMotion(
                pose, cfg, termType="CNT", termVal=100, spdType="mmSec", spd=900
            )
            cnt = 0
            while cnt < 1000 and controller.rmi.isLastLastSentMotionNotDone():
                controller.rmi.rmRecievePacket()
                cnt += 1

            rem = dt - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)
            else:
                print("LONG TIME:", abs(rem))
            

# =========================
# Helpers
# =========================
def reset_robot(controller: crx_controller, gripper: New_Gripper, init_joint=INIT_JOINT, speed=RESET_SPEED):
    controller.move_to_joint(init_joint, speed)
    if gripper is not None:
        # Use absolute-degree command via move_to_pos
        try:
            gripper.move_to_pos(int(GRIPPER_DEG_MAX))
        except Exception:
            # fallback if needed
            gripper.fully_open()
        time.sleep(gripper.get_operation_time())

def set_gripper_absolute_degree(gripper: New_Gripper, target_deg: float):
    """
    Move gripper to an absolute degree in [0, 1872] using move_to_pos()
    (preferred over relative open/close steps).
    """
    if gripper is None:
        return
    target = int(np.clip(target_deg, GRIPPER_DEG_MIN, GRIPPER_DEG_MAX))
    gripper.move_to_pos(target)

# =========================
# Rollout loop
# =========================
def run_policy_loop(env: RealEnv, controller: crx_controller, cfg, left_policy: DiffusionPolicy, right_policy: DiffusionPolicy, keys: KeyWatcher,
                    num_steps=NUM_STEPS, dt=DT, num_episode=1, replay=False):
    # policy.reset()
    left_policy.reset()
    right_policy.reset()
    print("Policy ready. Press Enter to start a rollout, 'r' to redo, 'q' to quit.")

    episode = 0
    while episode < num_episode:
        reset_robot(controller, env.gripper)
        env.exec_zero_actions(cfg, dt=dt)

        print(f"Press Enter to start rollout {episode}.")
        keys.wait_for("enter")

        step = 0
        # target6 = None
        current_state = None
        action_pred_error = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        while step < num_steps:
            t0 = time.time()
            obs, desired_action = env.get_obs(time_step=step, to_device=True, from_dataset=False)
            # obs_left, obs_right, desired_action = env.get_obs(time_step=step, to_device=True, from_dataset=False)
            action_list = []
            init_sample = torch.randn(
                size=(1, 32, 8),
                dtype=torch.float32,
                device="cuda"
            )
            if replay:
                act = desired_action
            else:
                with torch.inference_mode():
                    policy_st = time.time()
                    action = policy.select_action(obs)   # expected [dx,dy,dz,dW,dP,dR,grip_abs]
                    # def get_act(policy, obs, init_sample):
                    #     action = policy.select_action(obs, init_sample)
                    #     action_list.append(action)
                    #     # print("action_appended")
                    # thread1 = threading.Thread(target=get_act, args=(left_policy, obs_left, init_sample))
                    # thread2 = threading.Thread(target=get_act, args=(right_policy, obs_right, init_sample))
                    # thread1.start()
                    # thread2.start()
                    # thread1.join()
                    # thread2.join()
                    policy_et = time.time()
                    # action = (action_list[0] + action_list[1]) / 2.0
                    act = action.to("cpu").numpy()
                act = np.squeeze(act)
            print(f"time step {step} action: {act}")
            print(f"policy runtime:", policy_et - policy_st)
            # action_pred_error += np.abs(desired_action - act)

            # delta6   = np.array(act[:6], dtype=float)  # mm & deg deltas (world-frame)
            # dgrip = float(act[-1])                  # absolute deg [0..1872]
            
            # if target6 is None:
            #     target6, _ = controller.get_robot_state()
            # current6 = target6
            # target6 = compose_delta_next_pose_world(current6, delta6)

            # print("target pose:", target6)

            # controller.rmi.rmLinearMotion(
            #     target6, cfg, termType="CNT", termVal=100, spdType="mmSec", spd=900
            # )

            # set_gripper_absolute_degree(env.gripper, env.gripper.get_position_deg() + dgrip)

            if current_state is None:
                current_state, _ = controller.get_robot_state()

            spacemouse_state = spacemouse(act)

            if spacemouse_state.buttons[-1]:
                gripper.send_data(spacemouse_state)
                desired_state = current_state
            else:
                desired_state = get_desired_robot_state(spacemouse_state, current_state, SPEED)

            current_state = desired_state
            print("desired state:", desired_state)
            controller.rmi.rmLinearMotion(desired_state,cfg,termType='CNT',termVal=100,spdType="mmSec",spd=900)

            cnt = 0
            while cnt < 1000 and controller.rmi.isLastLastSentMotionNotDone():
                controller.rmi.rmRecievePacket()
                cnt += 1

            rem = dt - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)
            else:
                print("LONG TIME:", abs(rem))

            if keys.is_pressed("r"):
                print(f"Redoing rollout {episode}.")
                while keys.is_pressed("r"):
                    time.sleep(0.05)
                step = 0
                continue
            if keys.is_pressed("q"):
                print("Stopping deployment.")
                return

            step += 1

        episode += 1
        # print("action pred error: ", action_pred_error/step, np.mean(action_pred_error/step))


if __name__ == "__main__":
    trqutil = crxTrqUtil()
    trqutil.tool["position"] = np.array([0.0, 0.0, 0 * 138.6 / 1000])
    trqutil.tool["orientation"] = R.from_euler("ZYX", [0, 0, -180], degrees=True).as_quat()
    trqutil.LinkDynParam["Mass"] = np.zeros(6)
    trqutil.Ml = np.array(trqutil.LinkDynParam["Mass"])
    FTdata = multiprocessing.Array("f", np.zeros(6))
    pread = multiprocessing.Process(target=hspoRead, args=(FTdata, trqutil))
    pread.start()

    controller = connect_to_robot(FTdata, INIT_JOINT)
    gripper = New_Gripper(GRIPPER_SERIAL)
    # reset_robot(controller, gripper, init_joint=INIT_JOINT, speed=RESET_SPEED)
    Cpos,cfg,_,_= controller.rmi.rmGetCartPos()

    keys = KeyWatcher()
    env = RealEnv(controller, gripper)

    print("Loading policy…")
    policy = DiffusionPolicy.from_pretrained(PRETRAINED_POLICY)
    policy.to(torch.device("cuda"))
    # left_policy = DiffusionPolicy.from_pretrained(LEFT_PRETRAINED_POLICY)
    # right_policy = DiffusionPolicy.from_pretrained(RIGHT_PRETRAINED_POLICY)
    # left_policy.to(torch.device("cuda"))
    # right_policy.to(torch.device("cuda"))

    # policy.config.input_features['observation.images.fixed_left'] = policy.config.input_features['observation.images.fixed_left_right']
    # policy.config.image_features['observation.images.fixed_left'] = policy.config.image_features['observation.images.fixed_left_right']


    # print(policy.normalize_inputs)
    # exit()

    # policy.normalize_inputs = Normalize(policy.config.input_features, policy.config.normalization_mapping, dataset_stats)
    # policy.normalize_targets = Normalize(
    #     policy.config.output_features, policy.config.normalization_mapping, dataset_stats
    # )
    # policy.unnormalize_outputs = Unnormalize(
    #     policy.config.output_features, policy.config.normalization_mapping, dataset_stats
    # )



    print("Warming up get_obs forward…")
    for _ in range(10):
        _ = env.get_obs(time_step=0, to_device=True)

    print("Deploying policy.")
    run_policy_loop(env, controller, cfg, policy, keys, num_steps=NUM_STEPS, dt=DT, num_episode=1)
    # run_policy_loop(env, controller, cfg, left_policy, right_policy, keys, num_steps=NUM_STEPS, dt=DT, num_episode=1)

    cv2.destroyAllWindows()
    controller.close()
    pread.terminate()
