import numpy as np

def OTG(pd, pos, vel, Vm, U, dt):
    """ 1D 2nd order online trajectory generation """
    yn = pos - pd
    ynd = vel
    zn = (yn/dt+ynd/2.0)/(dt*U)
    znd = ynd/(dt*U)
    m = np.floor((1.0+np.sqrt(1.0+8.0*abs(zn)))/2.0)
    sigman = znd + zn/m+(m-1.0)/2.0*np.sign(zn)
    a = -U*np.clip(sigman,-1.0,1.0)*\
            (1+np.sign(
                ynd*np.sign(sigman)+Vm-dt*U
                ))/2.0
    vel_out = vel + a*dt
    pos_out = pos + vel*dt+0.5*a*dt**2
    return pos_out, vel_out