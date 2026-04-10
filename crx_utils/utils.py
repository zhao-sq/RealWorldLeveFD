import numpy as np

# read current robot joint position (fanuc convention, deg)
def readJntPos(rmi):
    Jpos,_=rmi.rmGetJntPos()
    jpos=np.array([Jpos['J1'],Jpos['J2'],Jpos['J3'],Jpos['J4'],Jpos['J5'],Jpos['J6']])
    return jpos

# read crx joint torque reading
def readJntTrq(pcdk):
    torque = np.zeros(6)
    for j in range(6):
        ls = pcdk.readSysVar('$CCC_GRP[1].$CUR_TRQ['+str(j+1)+']')
        torque[j] = float(ls[1]['value'])
    return torque

# read ext force from controller
def readExtF(pcdk):
    force = np.zeros(3)
    for j in range(3):
        ls = pcdk.readSysVar('$CCC_GRP[1].$CUR_FRC_VAL['+str(j+1)+']')
        force[j] = float(ls[1]['value'])
    return force

# move to joint position, with specific override, specify timeout for status feedback
# jpos=target joint position, (fanuc convention, deg)
# ovr=speed override, (integer, 1-100)
# timeout=maximum wait time before reporting error
def moveJnt(rmi,jpos,ovr,timeout=5):
    rmi.rmSetSpdOvd(ovr)
    rmi.rmJointMotionJRep(jpos)
    timeout0 = rmi.timeout
    rmi.settimeout(timeout)
    while rmi.isLastSentMotionNotDone():
        rmi.rmRecievePacket()
        # rmi.rmGetStatus()
    rmi.settimeout(timeout0)
