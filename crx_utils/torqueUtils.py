'''
CRX torque compensation utilities
Provide function to calculate gravity torque, set initial offset
for torque sensor reading
'''

import numpy as np
from scipy.spatial.transform import Rotation

class crxTrqUtil:
    def __init__(self):
        self.dof = 6
        self.DH={
            'Theta0':   np.array([0.,np.pi/2,0,0,0,0]),                                 # Joint variable offset (rad)
            'A':        np.array([0.,540,0,0,0,0])/1000.,                               # Joint offset (m)
            'D':        np.array([0.,0,0,-540,150,-160])/1000.,                         # Joint extension (m), adding base offset
            'Alpha':    np.array([np.pi/2,0,-np.pi/2,np.pi/2,-np.pi/2,np.pi]),          # Joint twist (rad)
        }
        # base = user frame?
        self.base={
            'position':     np.array([0.,0,0]),
            'orientation':  np.array([0.,0,0,1]),
        }
        # tool = tool frame
        self.tool={
            'position':     np.array([0.,0,0]),             # tool frame offset position [x,y,z], unit m, used typically for force sensor
            'orientation':  np.array([0.,0,0,1]),           # tool frame offset orientation [x,y,z,w], used typically for force sensor
        }
        # original link info
        self.LinkDynParam = {
            'Mass': np.array([10.799, 11.14, 4.434, 3.944, 2.577, 1.362]), # link mass (kg)
            'Cog': np.array([[-0.58,       -37.02,      45.65    ],
                            [-289.49,       0.05,	   183.43   ],
                            [0.04,        -10.81,  	   -35.16   ],
                            [0.14,        171.64,      1.75     ],
                            [1.28,         32.61,       -22.52  ],
                            [0.11,         -0.42,       -45.44  ]])/1000., # link cog (m) relative to each joint's frame
            'Inertia': np.array([[106721.703,  75747.82,    65877.281,	0,   0,   0],
                                [60053.57,    730636.313, 	701968.875,	0,   0,   0],
                                [18366.869,  	19883.949,	7608.320,	0,   0,   0],
                                [78972.359,	    4339.894,	78276.773,	0,   0,   0],
                                [7818.762,	    4530.349,	6158.803,	0,   0,   0],
                                [1450.347,	    1398.726,	1476.059,	0,   0,   0]])/1e6, # link inertia (diagonal) at cog (kgm^2)
        }
        self.Ml = np.array(self.LinkDynParam['Mass'])
        self.Rl = np.array(self.LinkDynParam['Cog'])
        self.linkInertia = np.array(self.LinkDynParam['Inertia'])
        # get link inertia matrix
        self.inertiaSetup()
        # gravity config
        self.gravity = np.array([0.,0,9.8])
        # update payload mass parameters
        self.Mp = 0.
        self.Rp = np.array([0.,0,0])
        self.payloadInertia = np.array([0.,0,0,0,0,0]) # payload inertia vector [Ixx Iyy Izz Ixy Iyz Ixz]
        self.payloadSetup()
        # offset jnttorque, rtb convention
        self.trqoffset = np.zeros(self.dof)

    # inertia matrix construction from vector [Ixx Iyy Izz Ixy Iyz Ixz]
    # x = inertia vector [Ixx Iyy Izz Ixy Iyz Ixz], output = inertia matrix
    def v2M(self,x):
        ret = np.array([[x[0], x[3], x[5]],
                        [x[3], x[1], x[4]],
                        [x[5], x[4], x[2]]])
        return ret

    # update link inertia variable
    def inertiaSetup(self):
        # set link inertia matrix
        self.Jlmat = np.zeros((self.dof,3,3))
        for i in range(self.dof):
            self.Jlmat[i] = self.v2M(self.linkInertia[i])

    # setup and update link inertia use payload info
    def payloadSetup(self):
        # get link info from backup
        self.Ml = np.array(self.LinkDynParam['Mass'])
        self.Rl = np.array(self.LinkDynParam['Cog'])
        self.linkInertia = np.array(self.LinkDynParam['Inertia'])
        self.inertiaSetup()
        # calculate final link inertia
        ml = self.Ml[-1] + self.Mp
        rl = (self.Rl[-1]*self.Ml[-1] + self.Rp*self.Mp)/ml
        jl_mat = self.Jlmat[-1]-self.Ml[-1]*(np.linalg.matrix_power(self.hat(rl-self.Rl[-1]),2))
        jl_mat += self.v2M(self.payloadInertia)-self.Mp*(np.linalg.matrix_power(self.hat(rl-self.Rp),2))
        # update
        self.Ml[-1] = ml
        self.Rl[-1] = rl
        self.Jlmat[-1] = jl_mat

    # update torque offset
    # trqoffset in rtb convention
    def settrqoffset(self,trqoffset):
        self.trqoffset = np.array(trqoffset)
        
    # update payload info
    # Mp = payload mass, Rp = payload cog in S6 frame (not tool frame), Inertia = payload inertia vector at cog ([Ixx Iyy Izz Ixy Iyz Ixz])
    def payloadUpdate(self,Mp,Rp,Inertia):
        # update payload info
        self.Mp = Mp
        self.Rp = np.array(Rp)
        self.payloadInertia = np.array(Inertia)
        # update link inertia
        self.payloadSetup()

    # angle convention transfer, either from fanuc to rtb, or rtb to fanuc
    def angCovt(self,angle):
        jpos = np.array(angle) # copy and transfer to np array
        j2,j3 = angle[1],angle[2] # get j2, j3
        jpos[1] = -j2 # j2'=-j2
        jpos[2] = j2+j3 # j3'=j2+j3
        return jpos

    # torque convention transfer, either from fanuc to rtb, or rtb to fanuc
    def torCovt(self,torque):
        tout = np.array(torque)
        tout[1] = -tout[1] # t2'=-t2
        return tout        

    # printout torque (debug purpose)
    def printTrq(self,trq):
        print(f'[{trq[0]:0.3f}, {trq[1]:0.3f}, {trq[2]:0.3f}, {trq[3]:0.3f}, {trq[4]:0.3f}, {trq[5]:0.3f}]')

    # calculate external force torque reading at tool frame (center)
    # orientation same as world frame
    # jntpos: robot joint position, fanuc convention (deg)
    # jnttrq: crx joint torque sensor reading (Nm), fanuc convention
    def extFT(self,jnttrq,jntpos):
        q = self.angCovt(jntpos)/180.*np.pi # rtb joint position (rad)
        trq = self.torCovt(jnttrq) # rtb joint torque (Nm)
        trq_g = self.gra_s(q) # gravity torque, rtb convention
        trq_off = self.trqoffset # torque offset, rtb convention
        J,_,_ = self.jacobian_tcp(q) # jacobian
        trq -= trq_g + trq_off # remove gravity torque and torque offset
        F = -(np.linalg.pinv(J.T)@trq) # external force torque applied to robot, F = [f,t], f=3D force, t=3D torque
        return F
    
    # calculate external force torque reading at tool frame (center)
    # orientation same as world frame
    # jntpos: robot joint position, fanuc convention (deg)
    # jnttrq: crx joint torque sensor reading (Nm), fanuc convention
    def extFT_simple(self,jnttrq,jntpos):
        q = self.angCovt(jntpos)/180.*np.pi # rtb joint position (rad)
        trq = self.torCovt(jnttrq) # rtb joint torque (Nm)
        trq_off = self.trqoffset # torque offset, rtb convention
        J,_,_ = self.jacobian_tcp(q) # jacobian
        trq -= trq_off # remove gravity torque and torque offset
        F = -(np.linalg.pinv(J.T)@trq) # external force torque applied to robot, F = [f,t], f=3D force, t=3D torque
        return F
    
    # calculate external force torque reading at tool frame (center)
    # orientation as tool frame
    # jntpos: robot joint position, fanuc convention (deg)
    # jnttrq: crx joint torque sensor reading (Nm), fanuc convention
    def extFT_ee_simple(self,jnttrq,jntpos):
        q = self.angCovt(jntpos)/180.*np.pi # rtb joint position (rad)
        trq = self.torCovt(jnttrq) # rtb joint torque (Nm)
        trq_off = self.trqoffset # torque offset, rtb convention
        J,_,_ = self.jacobian_ee_tcp(q) # jacobian
        trq -= trq_off # remove gravity torque and torque offset
        F = -(np.linalg.pinv(J.T)@trq) # external force torque applied to robot, F = [f,t], f=3D force, t=3D torque
        return F
    
    # calculate external force torque reading at tool frame (center)
    # orientation as tool frame
    # jntpos: robot joint position, fanuc convention (deg)
    # jnttrq: crx joint torque sensor reading (Nm), fanuc convention
    def extFT_ee(self,jnttrq,jntpos):
        q = self.angCovt(jntpos)/180.*np.pi # rtb joint position (rad)
        trq = self.torCovt(jnttrq) # rtb joint torque (Nm)
        trq_g = self.gra_s(q) # gravity torque, rtb convention
        trq_off = self.trqoffset # torque offset, rtb convention
        J,_,_ = self.jacobian_ee_tcp(q) # jacobian
        trq -= trq_g + trq_off # remove gravity torque and torque offset
        F = -(np.linalg.pinv(J.T)@trq) # external force torque applied to robot, F = [f,t], f=3D force, t=3D torque
        return F
    
    # payload and trqoffset calibration and update
    # !!! make sure tool offset is setted to all zero, payload is setted to all zero
    # requires multiple measurements, each include jnt position and jnttorque
    # jnttrq = list of joint torque measurements (fanuc convention) (Nm)
    # jntpos = list of joint position measurements (fanuc convention) (deg)
    # jac_thrd = threshold of Jacobian determinent for data measurement
    # returned torque offset in rtb convention
    def calibrate_payload_offset(self,jnttrq,jntpos,jac_thrd):
        print('Start calibration. Make sure tool offset is all 0!')
        print(f' Current tool offset: {self.tool}')
        print(f' Current payload: Mp {self.Mp}, Rp {self.Rp}, Ip {self.payloadInertia}')
        # get number of joint torque measurements
        n = len(jnttrq)
        if n!=len(jntpos):
            print('torque and position data number mismatch!')
            return None, None
        G=self.gravity.copy()
        G_hat = self.hat(G)
        G_mat=np.array([G]).T
        K_mat = []
        T_mat = []
        for j in range(n):
            q=self.angCovt(jntpos[j])/180.*np.pi # joint position, rbt convention, rad
            J,_,orn = self.jacobian_tcp(q)
            if abs(np.linalg.det(J))<jac_thrd:
                print(f'measurement {j} close to singualirity!')
                return None, None
            t_gravity=self.gra_s(q) # robot link gravity torque
            t_gravity = np.round(t_gravity, decimals=3) # match pcdk reading (pcdk only shows 3 decimals)
            R = Rotation.from_quat(orn).as_matrix()
            a=np.concatenate([-G_mat,np.zeros((3,3))],axis=1)
            b=np.concatenate([np.zeros((3,1)),G_hat@R],axis=1)
            c = -( J.T@np.concatenate([a,b]) )
            K_mat.append( np.concatenate((np.eye(self.dof),c),axis=1) )
            T_mat.append(self.torCovt(jnttrq[j])-t_gravity)
        K_mat = np.concatenate(K_mat)
        T_mat = np.concatenate(T_mat)
        print('===============================')
        print(f'Start calibration. \n \t Data points: {n} \n \t Calibration matrix cond: {np.linalg.cond(K_mat)} \n \t Calibration matrix rank: {np.linalg.matrix_rank(K_mat)}')
        x = np.linalg.pinv(K_mat)@T_mat
        trq_offset = np.array(x[:self.dof])
        Mp = x[self.dof]
        if abs(Mp)>0:
            Rp = np.array(x[-3:])/Mp
        else:
            print('Payload mass close to 0!')
            Rp = np.zeros(3)
        return Mp, Rp, trq_offset, K_mat, T_mat

    
    # calculate joint torque reading offset (rtb convention)
    # jntpos: robot joint position, fanuc convention (deg)
    # jnttrq: crx joint torque sensor reading (Nm), fanuc convention
    def calTrqOffset(self,jnttrq,jntpos):
        q = self.angCovt(jntpos)/180.*np.pi # rtb joint position (rad)
        trq = self.torCovt(jnttrq) # rtb joint torque (Nm)
        trq_g = self.gra_s(q) # gravity torque, rtb convention
        trq_off = trq-trq_g # offset torque, rtb convention
        return trq_off
    
    # calculate joint torque reading offset (rtb convention)
    # jntpos: robot joint position, fanuc convention (deg)
    # jnttrq: crx joint torque sensor reading (Nm), fanuc convention
    def calTrqOffset_simple(self,jnttrq):
        trq_off = self.torCovt(jnttrq) # rtb joint torque (Nm)
        return trq_off
        
    def forwardKinematics(self,jntpos):
        # forward kinematics getting tool frame, return position and orientation (in quaternion)
        A=self.DH['A']
        D=self.DH['D']
        alpha=self.DH['Alpha']
        theta0=self.DH['Theta0']
        R_tool=Rotation.from_quat(self.tool['orientation']).as_matrix()
        T_tool=np.array(self.tool['position'])
        Trans_tool = np.concatenate((R_tool,T_tool.reshape(-1,1)),1)
        Trans_tool = np.concatenate((Trans_tool,np.array([[0,0,0,1]])),0)
        R_base=Rotation.from_quat(self.base['orientation']).as_matrix()
        T_base=np.array(self.base['position'])
        Trans_base = np.concatenate((R_base,T_base.reshape(-1,1)),1)
        Trans_base = np.concatenate((Trans_base,np.array([[0,0,0,1]])),0)
        # Kinematics
        q=np.array(jntpos) + theta0
        # manully loop for each joint
        TCP_T=Trans_base
        for i in range(self.dof):
            TCP_T=TCP_T @ \
                np.array([[np.cos(q[i]), -np.sin(q[i])*np.cos(alpha[i]),  np.sin(q[i])*np.sin(alpha[i]), A[i]*np.cos(q[i])],
                          [np.sin(q[i]),  np.cos(q[i])*np.cos(alpha[i]), -np.cos(q[i])*np.sin(alpha[i]), A[i]*np.sin(q[i])],
                          [0,             np.sin(alpha[i]),               np.cos(alpha[i]),              D[i]],
                          [0,                0,                           0,               1]])
        # up to now, forward kinematics, S0 to S6
        TCP_T=TCP_T@Trans_tool # S6 to tool
        TCP=TCP_T[0:3,3] # TCP point in the world frame
        orientation=Rotation.from_matrix(TCP_T[0:3,0:3])
        orientation=orientation.as_quat() # TCP frame orientation relative to world frame
        return TCP, orientation
    
    def jacobian_tcp(self,jntpos):
        # get jacobian located at tool frame, orientation as world frame
        # also forward kinematics for TCP position and orientation (in quaternion)
        A=self.DH['A']
        D=self.DH['D']
        alpha=self.DH['Alpha']
        theta0=self.DH['Theta0']
        R_tool=Rotation.from_quat(self.tool['orientation']).as_matrix()
        T_tool=np.array(self.tool['position'])
        Trans_tool = np.concatenate((R_tool,T_tool.reshape(-1,1)),1)
        Trans_tool = np.concatenate((Trans_tool,np.array([[0,0,0,1]])),0)
        R_base=Rotation.from_quat(self.base['orientation']).as_matrix()
        T_base=np.array(self.base['position'])
        Trans_base = np.concatenate((R_base,T_base.reshape(-1,1)),1)
        Trans_base = np.concatenate((Trans_base,np.array([[0,0,0,1]])),0)
        # Kinematics
        q=np.array(jntpos)+theta0
        # manully loop for each joint
        w=np.zeros((3,6))
        r_0=np.zeros((3,7))
        J=np.zeros((6,6)) # jacobian
        TCP_T=Trans_base
        for i in range(6):
            w[:,i] = TCP_T[0:3,2]
            r_0[:,i] = TCP_T[0:3,3]
            TCP_T=TCP_T @ \
                np.array([[np.cos(q[i]), -np.sin(q[i])*np.cos(alpha[i]),  np.sin(q[i])*np.sin(alpha[i]), A[i]*np.cos(q[i])],
                          [np.sin(q[i]),  np.cos(q[i])*np.cos(alpha[i]), -np.cos(q[i])*np.sin(alpha[i]), A[i]*np.sin(q[i])],
                          [0,             np.sin(alpha[i]),               np.cos(alpha[i]),              D[i]],
                          [0,                0,                           0,               1]])
        r_0[:,6]=TCP_T[0:3,3]
        # up to now, forward kinematics, S0 to S6
        TCP_T=TCP_T@Trans_tool # S6 to tool
        TCP=TCP_T[0:3,3] #orig_abs;% TCP point in the world frame
        for i in range(6):
            J[:,i]=np.concatenate(
                (np.cross(r_0[:,i]-TCP,w[:,i]),w[:,i])
            )
        orientation=Rotation.from_matrix(TCP_T[0:3,0:3])
        orientation=orientation.as_quat() # TCP frame orientation relative to world frame
        return J, TCP, orientation
    
    def jacobian_ee_tcp(self,jntpos):
        # get jacobian relative to end effector (tcp) frame
        # orientation as tool frame
        A=self.DH['A']
        D=self.DH['D']
        alpha=self.DH['Alpha']
        theta0=self.DH['Theta0']
        R_tool=Rotation.from_quat(self.tool['orientation']).as_matrix()
        T_tool=np.array(self.tool['position'])
        Trans_tool = np.concatenate((R_tool,T_tool.reshape(-1,1)),1)
        Trans_tool = np.concatenate((Trans_tool,np.array([[0,0,0,1]])),0)
        R_base=Rotation.from_quat(self.base['orientation']).as_matrix()
        T_base=np.array(self.base['position'])
        Trans_base = np.concatenate((R_base,T_base.reshape(-1,1)),1)
        Trans_base = np.concatenate((Trans_base,np.array([[0,0,0,1]])),0)
        # Kinematics
        q=np.array(jntpos)+theta0
        # manully loop for each joint
        w=np.zeros((3,6))
        r_0=np.zeros((3,7))
        J=np.zeros((6,6)) # jacobian
        TCP_T=Trans_base
        for i in range(6):
            w[:,i] = TCP_T[0:3,2]
            r_0[:,i] = TCP_T[0:3,3]
            TCP_T=TCP_T @ \
                np.array([[np.cos(q[i]), -np.sin(q[i])*np.cos(alpha[i]),  np.sin(q[i])*np.sin(alpha[i]), A[i]*np.cos(q[i])],
                          [np.sin(q[i]),  np.cos(q[i])*np.cos(alpha[i]), -np.cos(q[i])*np.sin(alpha[i]), A[i]*np.sin(q[i])],
                          [0,             np.sin(alpha[i]),               np.cos(alpha[i]),              D[i]],
                          [0,                0,                           0,               1]])
        r_0[:,6]=TCP_T[0:3,3]
        # up to now, forward kinematics, S0 to S6
        TCP_T=TCP_T@Trans_tool # S6 to tool
        TCP=TCP_T[0:3,3] #orig_abs;% TCP point in the world frame
        TCP_R = TCP_T[0:3,0:3] # tool frame orientation
        for i in range(6):
            s = TCP_R.T @ w[:,i] # change orientation
            r = TCP_R.T @ (r_0[:,i]-TCP) # change orientation
            J[:,i]=np.concatenate(
                (np.cross(r,s),s)
            )
        orientation=Rotation.from_matrix(TCP_T[0:3,0:3])
        orientation=orientation.as_quat() # TCP frame orientation relative to world frame
        return J, TCP, orientation
    
    # joint transformation for motion
    def jcalc(self, j, q):
        c = np.cos(q)
        s = np.sin(q)
        ca = np.cos(self.DH['Alpha'][j])
        sa = np.sin(self.DH['Alpha'][j])
        E = np.array([[c,   -s*ca,  s*sa    ],
                      [s,   c*ca,   -c*sa   ],
                      [0,   sa,     ca      ]]).T # E = R.'
        r = np.array([ self.DH['A'][j]*c, self.DH['A'][j]*s, self.DH['D'][j] ])
        # assemble result
        a=np.concatenate((E, np.zeros((3,3))), axis=1)
        b=np.concatenate((-E@self.hat(r), E), axis=1)
        XJ = np.concatenate((a,b))
        return XJ
        
    # vector to matrix for cross product
    def hat(self, w):
        ret = np.array([[0.,      -w[2],   w[1]],
                        [w[2],      0,     -w[0]],
                        [-w[1],     w[0],    0]])
        return ret
    
    # spatial inertia matrix
    def mcI(self, m, c, I):
        C = self.hat(c)
        a = np.concatenate((I + m*(C@C.T), m*C), axis=1)
        b = np.concatenate((m*C.T, m*np.eye(3)), axis=1)
        ret = np.concatenate((a,b))
        return ret
    
    # calculate gravity torque using ABA
    def gra_s(self, jntpos):
        # initialize torque output
        Torque = np.zeros(self.dof)
        gravity = np.concatenate((np.zeros(3),self.gravity))
        theta0=self.DH['Theta0']
        q=np.array(jntpos)+theta0
        Si = np.array([0.,0,1,0,0,0])
        # temp vectors
        Xup = [0]*self.dof
        a = [0]*self.dof
        f = [0]*self.dof
        # forward recursion
        for j in range(self.dof):
            Xup[j] = self.jcalc(j,q[j])
            if j==0:
                a[j] = Xup[j]@gravity
            else:
                a[j] = Xup[j]@a[j-1]
            I = self.mcI(self.Ml[j],self.Rl[j],self.Jlmat[j])
            f[j] = I@a[j]
        # backward recursion
        for j in range(self.dof-1,-1,-1):
            Torque[j] = np.dot(Xup[j]@Si, f[j])
            if j>0:
                f[j-1] += Xup[j].T@f[j]
        # return gravity torque
        return Torque