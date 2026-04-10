import numpy as np
from crx_utils import pyRMI
import time
import pygame
from scipy.spatial.transform import Rotation

# sample xbox controller keymap, js=joystick handle, rmi=rmi interface
# linux version
def keymap_xbox(js,rmi:pyRMI,config,thrd=0.5,grip_thrd=0.5):
    exit_flg = False
    for event in pygame.event.get():
        if event.type == pygame.JOYDEVICEREMOVED:
            print('joystick removed\n')
            exit_flg = True
        elif event.type == pygame.JOYBUTTONDOWN and event.button == 3: # x key
            # print(pygame.event.event_name(event.type))
            print('exit\n')
            exit_flg=True
        elif event.type == pygame.JOYBUTTONDOWN:
            if event.button == 7: # 'RB'
                vel0=config['velLin']
                config['velLin']=min([config['velLin_max'],config['velLin']+config['velLin_delta']])
                vel1=config['velLin']
                #print(f'Linvel changed from {vel0} to {vel1}')
            elif event.button == 6: # 'LB'
                vel0=config['velLin']
                config['velLin']=max([0,config['velLin']-config['velLin_delta']])
                vel1=config['velLin']
                #print(f'Linvel changed from {vel0} to {vel1}')
            elif event.button == 11: # 'start'
                vel0=config['velRot']
                config['velRot']=min([config['velRot_max'],config['velRot']+config['velRot_delta']])
                vel1=config['velRot']
                #print(f'Rotvel changed from {vel0} to {vel1}')
            elif event.button == 10: # 'back'
                vel0=config['velRot']
                config['velRot']=max([0,config['velRot']-config['velRot_delta']])
                vel1=config['velRot']
                #print(f'Rotvel changed from {vel0} to {vel1}')
            elif event.button == 0: # 'A'
                Cpos,cfg,_=rmi.rmGetCartPos()
                print(f'Current cart pos: \t{Cpos}')
            elif event.button == 4: # 'Y'
                vellin=config['velLin']
                velrot=config['velRot']
                print(f'Linvel: {vellin}, rotvel: {velrot}')
    pygame.event.clear()
    # get gripper motion
    grip_open, grip_close = js.get_axis(4), js.get_axis(5)
    grip_open = (grip_open>grip_thrd)
    grip_close = (grip_close>grip_thrd)
    if grip_open:
        grip_close = False
    if grip_close:
        grip_open = False
    # get xyz wpr motion
    ax,ay =  js.get_axis(1),js.get_axis(0)
    aw,ap = -js.get_axis(3),js.get_axis(2)
    ar,az =  js.get_hat(0)
    aw = -aw
    action = np.array([ax,ay,az,aw,ap,ar],dtype=np.float32)
    # set threshold
    for i in range(len(action)):
        if np.abs(action[i])<thrd:
            action[i]=np.float32(0.)
        else:
            action[i]=np.float32(np.sign(action[i]))
    action[0:3]*=config['velLin']
    action[3:6]*=config['velRot']
    cmd={'action':action,'exit_flg':exit_flg,'grip_open':grip_open,'grip_close':grip_close}
    return cmd
# windows version
# def keymap_xbox(js,rmi:pyRMI,config,thrd=0.5,grip_thrd=0.5):
#     exit_flg = False
#     for event in pygame.event.get():
#         if event.type == pygame.JOYDEVICEREMOVED:
#             print('joystick removed\n')
#             exit_flg = True
#         elif event.type == pygame.JOYBUTTONDOWN and event.button == 2:
#             # print(pygame.event.event_name(event.type))
#             print('exit\n')
#             exit_flg=True
#         elif event.type == pygame.JOYBUTTONDOWN:
#             if event.button == 5: # 'RB'
#                 vel0=config['velLin']
#                 config['velLin']=min([config['velLin_max'],config['velLin']+config['velLin_delta']])
#                 vel1=config['velLin']
#                 #print(f'Linvel changed from {vel0} to {vel1}')
#             elif event.button == 4: # 'LB'
#                 vel0=config['velLin']
#                 config['velLin']=max([0,config['velLin']-config['velLin_delta']])
#                 vel1=config['velLin']
#                 #print(f'Linvel changed from {vel0} to {vel1}')
#             elif event.button == 7: # 'start'
#                 vel0=config['velRot']
#                 config['velRot']=min([config['velRot_max'],config['velRot']+config['velRot_delta']])
#                 vel1=config['velRot']
#                 #print(f'Rotvel changed from {vel0} to {vel1}')
#             elif event.button == 6: # 'back'
#                 vel0=config['velRot']
#                 config['velRot']=max([0,config['velRot']-config['velRot_delta']])
#                 vel1=config['velRot']
#                 #print(f'Rotvel changed from {vel0} to {vel1}')
#             elif event.button == 0: # 'A'
#                 Cpos,cfg,_=rmi.rmGetCartPos()
#                 print(f'Current cart pos: \t{Cpos}')
#             elif event.button == 3: # 'Y'
#                 vellin=config['velLin']
#                 velrot=config['velRot']
#                 print(f'Linvel: {vellin}, rotvel: {velrot}')
#     pygame.event.clear()
#     # get gripper motion
#     grip_open, grip_close = js.get_axis(5), js.get_axis(4)
#     grip_open = (grip_open>grip_thrd)
#     grip_close = (grip_close>grip_thrd)
#     if grip_open:
#         grip_close = False
#     if grip_close:
#         grip_open = False
#     # get xyz wpr motion
#     ax,ay =  js.get_axis(1),js.get_axis(0)
#     aw,ap = -js.get_axis(2),js.get_axis(3)
#     ar,az =  js.get_hat(0)
#     ar = -ar
#     action = np.array([ax,ay,az,aw,ap,ar],dtype=np.float32)
#     # set threshold
#     for i in range(len(action)):
#         if np.abs(action[i])<thrd:
#             action[i]=np.float32(0.)
#         else:
#             action[i]=np.float32(np.sign(action[i]))
#     action[0:3]*=config['velLin']
#     action[3:6]*=config['velRot']
#     cmd={'action':action,'exit_flg':exit_flg,'grip_open':grip_open,'grip_close':grip_close}
#     return cmd

# sample teleoperation function, rmi=rmi interface, keymapfcn=joystick key map
# return robot position and time record during operation as np array, time set to start from 0 (assume robot tick 2ms for crx)
# loopsleep: in pygame loop whether wait 1/frame_rate time, default is True, could disable if motion is not smooth
# safeLbnd and safeUbnd: safety lower and upper bound of X,Y,Z,W,P,R value. if None, no bounding
def teleoperate(rmi:pyRMI,keymapfcn=keymap_xbox,frame_rate=30,loopsleep=True,safeLbnd:np.array=None,safeUbnd:np.array=None):
    posrec=[]
    timerec=[]
    # config setting
    config={'cntlimit':10000}
    config['velLin'] = 100 # initial linear velocity: 100 (target unit mm/s, not exactly robot speed)
    config['velLin_delta'] = 10 # linvel adjustment delta
    config['velLin_max'] = 1000 # maximum allowed linvel setting
    config['velRot'] = 30 # initial angular velocity: 10 (target unit deg/s, not exactly robot speed)
    config['velRot_delta'] = 3 # rotvel adjustment delta
    config['velRot_max'] = 90 # maximum allowed rotvel setting

    # get initial pose and display
    rmi.rmReset()
    time.sleep(0.1)
    rmi.rmGetStatus()
    cnt=0
    while cnt<config['cntlimit'] and rmi.isLastSentMotionNotDone():
        # rmi.rmGetStatus()
        rmi.rmRecievePacket()
        cnt+=1
    time.sleep(0.1)
    Cpos,cfg,robtime=rmi.rmGetCartPos()
    timerec.append(robtime)
    posrec.append(np.array([Cpos['X'],Cpos['Y'],Cpos['Z'],Cpos['W'],Cpos['P'],Cpos['R']]))
    print(f'Initial pose: {Cpos}')
    print(f'Configuration: {cfg}')
    print(' ')
    vellin,velrot=config['velLin'],config['velRot']
    print(f'Initial linvel: {vellin}, initial rotvel: {velrot}')
    vellindelta,velrotdelta=config['velLin_delta'],config['velRot_delta']
    print(f'Linvel delta: {vellindelta}, rotvel delta: {velrotdelta}')
    vellinmax,velrotmax=config['velLin_max'],config['velRot_max']
    print(f'Linvel max: {vellinmax}, rotvel max: {velrotmax}')
    print(' ')

    # initialize joystick control
    pygame.init()
    pygame.event.clear()
    pygame.joystick.init()
    clock = pygame.time.Clock()
    if pygame.joystick.get_count()<1:
        print('Cannot detect joystick')
        posrec=np.array(posrec)
        timerec=np.array(timerec)*0.002
        timerec-=timerec[0]
        return posrec,timerec
    js = pygame.joystick.Joystick(0)
    print(f'joystick name: {js.get_name()}')
    print(f'joystick initialized: {js.get_init()}')
    print('Ready for teleoperation ......')
    print(' ')
    
    # drive robot
    exit_flg = False
    # print(exit_flg)
    dt = 1/frame_rate
    poscmd=np.array([Cpos['X'],Cpos['Y'],Cpos['Z'],Cpos['W'],Cpos['P'],Cpos['R']])
    
    while not exit_flg:
        if loopsleep:
            _=clock.tick(frame_rate)
        cmds = keymapfcn(js,rmi,config)
        exit_flg = cmds['exit_flg']
        grip_open,grip_close = cmds['grip_open'],cmds['grip_close']
        if not rmi.ServoReady:
            exit_flg=True
        if not exit_flg:
            # drive robot
            velcmd = cmds['action']
            poscmd[0:3] += dt*velcmd[0:3]
            rot0=Rotation.from_euler('ZYX',[poscmd[5],poscmd[4],poscmd[3]],degrees=True)
            drot=Rotation.from_euler('xyz',[dt*velcmd[3],dt*velcmd[4],dt*velcmd[5]],degrees=True)
            poscmd[3:6] = (drot*rot0).as_euler('xyz',degrees=True)
            if (safeLbnd is not None) and (safeUbnd is not None):
                poscmd = np.clip(poscmd, np.array(safeLbnd), np.array(safeUbnd))
            if grip_open or grip_close:
                rmi.rmLinearMotion(poscmd,cfg,termType='FINE',spdType = "mSec", spd = int(dt*1000))
                cnt=0
                while cnt<config['cntlimit'] and rmi.isLastSentMotionNotDone():
                    # rmi.rmGetStatus()
                    rmi.rmRecievePacket()
                    Cpos,cfg,robtime=rmi.rmGetCartPos()
                    timerec.append(robtime)
                    posrec.append(np.array([Cpos['X'],Cpos['Y'],Cpos['Z'],Cpos['W'],Cpos['P'],Cpos['R']]))
                    cnt+=1
                if grip_open:
                    rmi.rmCall('FG7_OPEN')
                else:
                    rmi.rmCall('FG7_CLOSE')
                rmi.confirmCall()
            else:
                rmi.rmLinearMotion(poscmd,cfg,termType='CNT',termVal=100,spdType = "mSec", spd = int(dt*1000))
                cnt=0
                while cnt<config['cntlimit'] and rmi.isLastLastSentMotionNotDone():
                    # rmi.rmGetStatus()
                    rmi.rmRecievePacket()
                    Cpos,cfg,robtime=rmi.rmGetCartPos()
                    timerec.append(robtime)
                    posrec.append(np.array([Cpos['X'],Cpos['Y'],Cpos['Z'],Cpos['W'],Cpos['P'],Cpos['R']]))
                    cnt+=1
        else:
            if not rmi.ServoReady:
                print('Exit due to error')
                posrec=np.array(posrec)
                timerec=np.array(timerec)*0.002
                timerec-=timerec[0]
                return posrec,timerec
            rmi.rmLinearMotion(poscmd,cfg)
            cnt=0
            while cnt<config['cntlimit'] and rmi.isLastSentMotionNotDone():
                rmi.rmRecievePacket()
                cnt+=1
        
    print('Exiting teleoperation ......')
    time.sleep(0.1)
    Cpos,cfg,_=rmi.rmGetCartPos()
    print(f'Exiting pose: \n \t{Cpos}')
    posrec=np.array(posrec)
    timerec=np.array(timerec)*0.002
    timerec-=timerec[0]
    return posrec,timerec
