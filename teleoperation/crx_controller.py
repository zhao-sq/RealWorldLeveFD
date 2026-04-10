import numpy as np
import time
from crx_utils import pyRMI, moveJnt

class crx_controller():
    def __init__(self, FT_data, ip='192.168.1.101'):
        self.ip = ip
        self.FT_data = FT_data
        # end effector 6 dimensional
        self.rmi=pyRMI(ip) # rmi
        self.rmi.initializRMI()
        self.rmi.rmSetSpdOvd(100)
        " safety ('ON' = turn off safety, 'OFF' = turn on safety) "
        # self.rmi.rmWriteDOUT(70,'ON')
        self.rmi.rmWriteDOUT(70,'OFF')
        " parameter definition "
        self.frame_rate=30
        self.smooth=True #False
        # init poses
        self.init_pull_pose = np.array([-12.832,-1.872,-15.172,0,-74.828,102])#previous
        # self.init_pull_pose = np.array([-11.952,6.352,-15.764,0,-74.241,91.707])
        # self.init_pull_pose = np.array([-14.941,1.616,-15.00,0,-75.0,12.941])

        # # safe bounds
        # Cpos,cfg,_=self.rmi.rmGetCartPos()
        # pos=np.array([Cpos['X'],Cpos['Y'],Cpos['Z'],Cpos['W'],Cpos['P'],Cpos['R']])
        pos = np.array([5.400e+02, -1.500e+02,  2.414e+02,  1.434e-05,  0.000e+00,  9.000e+01])
        self.safeLbnd=pos+[-300,-500,-750,-30,-30,-178-88]
        self.safeUbnd=pos+[ 500, 500, 100, 30, 30, 88]
        self.force_limit = 20

        # robot is on the left side or not
        self.robot_fixture_location = None

        # MPC settings
        self.cable_fix_point = np.array([455, -282.8, 101.4])

        # buffer
        self.force_buffer = np.zeros(6)
        Cpos,cfg,_,_=self.rmi.rmGetCartPos()
        robot_pose = np.array([Cpos['X'],Cpos['Y'],Cpos['Z'],Cpos['W'],Cpos['P'],Cpos['R']])
        self.eef_pose = robot_pose

        # plotting setting
        self.verbose = False

        # init
        self.rmi.rmReset()
        time.sleep(0.1)
        self.rmi.rmGetStatus()
        cnt=0
        while cnt<10000 and self.rmi.isLastSentMotionNotDone():
            # rmi.rmGetStatus()
            self.rmi.rmRecievePacket()
            cnt+=1
    
    def move_to_joint(self, joint, speed):
        moveJnt(self.rmi,joint.tolist(),speed)

    def move_to_crt(self,cpos, speed):
        Cpos,cfg,_,_=self.rmi.rmGetCartPos()

        self.rmi.rmLinearMotion(cpos,cfg,termType='FINE',termVal=100,spd = speed)
        cnt=0
        while cnt<10000 and self.rmi.isLastSentMotionNotDone():
            # rmi.rmGetStatus()
            self.rmi.rmRecievePacket()
            # Cpos,cfg,robtime=self.rmi.rmGetCartPos()
            cnt+=1
        self.get_robot_state()

    def get_robot_state(self):
        # self.rmi.rmRecievePacket()
        Cpos,cfg,_,response=self.rmi.rmGetCartPos()
        robot_pose = np.array([Cpos['X'],Cpos['Y'],Cpos['Z'],Cpos['W'],Cpos['P'],Cpos['R']])
        with self.FT_data.get_lock():
            F_ext_ee=np.frombuffer(self.FT_data.get_obj(),np.float32)
        self.force_buffer = F_ext_ee
        # print(koopman_state)
        return robot_pose, response

    def switch_origin_check(self, next_fix_point):
        Cpos,cfg,_=self.rmi.rmGetCartPos()
        robot_pose = np.array([Cpos['X'],Cpos['Y'],Cpos['Z'],Cpos['W'],Cpos['P'],Cpos['R']])
        if np.linalg.norm(robot_pose[:2] - self.cable_fix_point[:2])/1000 >= np.linalg.norm(next_fix_point[:2] - self.cable_fix_point[:2])/1000:
            curr_to_next_fix = np.zeros(3)
            curr_to_next_fix[:2] = (next_fix_point[:2] - self.cable_fix_point[:2])/1000
            next_to_robot = np.zeros(3)
            next_to_robot[:2] = (robot_pose[:2] - self.cable_fix_point[:2])/1000
            curr_robot_fixture_location = np.sign(np.cross(curr_to_next_fix, next_to_robot)[-1])
            if self.robot_fixture_location == None:
                self.robot_fixture_location = curr_robot_fixture_location
                return False
            elif curr_robot_fixture_location != self.robot_fixture_location:
                self.robot_fixture_location = None
                return True
            else:
                return False
        else:
            return False

    
    def gripper_motion(self, close):
        if close:
            self.rmi.rmCall('FG7_CLOSE')
        else:
            self.rmi.rmCall('FG7_OPEN')

    def move_delta_cart_pos(self, velcmd):
        # get current pose
        Cpos,cfg,_=self.rmi.rmGetCartPos()
        with self.FT_data.get_lock():
            F_ext_ee=np.frombuffer(self.FT_data.get_obj(),np.float32)
        " adding force / torque safety "
        for i in range(3):
            f=F_ext_ee[i]
            if f*velcmd[i]<0 and abs(f)>self.force_limit:
                velcmd[i]=0
        dt = 1/self.frame_rate
        poscmd=np.array([Cpos['X'],Cpos['Y'],Cpos['Z'],Cpos['W'],Cpos['P'],Cpos['R']])
        # add delta pose
        poscmd[0:3] = np.array([Cpos['X'],Cpos['Y'],Cpos['Z']]) + dt*velcmd[0:3]*5
        poscmd[3:6] = np.array([Cpos['W'],Cpos['P'],Cpos['R']]) + dt*velcmd[3:6]*5
        # clip
        if (self.safeLbnd is not None) and (self.safeUbnd is not None):
            poscmd = np.clip(poscmd, np.array(self.safeLbnd), np.array(self.safeUbnd))
            print('here?')
            
        # drive
        # if self.smooth:
        #     self.rmi.rmLinearMotion(poscmd,cfg,termType='FINE',spdType = "mSec", spd = int(dt*1000))
        # else:
        #     self.rmi.rmLinearMotion(poscmd,cfg,termType='FINE',termVal=100)
        
        if self.smooth:
            self.rmi.rmLinearMotion(poscmd,cfg,termType='CNT',termVal=100,spdType = "mSec", spd = int(dt*1000))
        else:
            self.rmi.rmLinearMotion(poscmd,cfg,termType='CNT',termVal=100)
        if self.smooth:
            self.rmi.rmLinearMotion(poscmd,cfg,termType='CNT',termVal=100,spdType = "mSec", spd = int(dt*1000))
        else:
            self.rmi.rmLinearMotion(poscmd,cfg,termType='CNT',termVal=100)
        
        cnt=0
        while cnt<10000 and self.rmi.isLastLastSentMotionNotDone():
            # rmi.rmGetStatus()
            self.rmi.rmRecievePacket()
            # Cpos,cfg,robtime=self.rmi.rmGetCartPos()
            cnt+=1
        self.get_robot_state()

    
    def close(self):
        self.rmi.closeRMI()