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
                config['start_recording'] = not config['start_recording']
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
    cmd={'action':action,'exit_flg':exit_flg,'grip_open':grip_open,'grip_close':grip_close, 'start_recording':config['start_recording']}
    return cmd