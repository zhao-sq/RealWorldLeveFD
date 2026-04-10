'''
Python Remote Motion Interface (pyRMI)
Based on FANUC America Offical Manual - Remote Motion Interface (MAROGRMIM11191E REV A) and previous implmentation (pyRemoteMotion)
Requirement:
Robot Controller Version: 9.10 and later
Software Option: R912 - Remote Motion Interface (Not Compatible with R892 - PLC Motion Interface)
Communication: Ethernet connection
Limitation:
1. Support only one group motion
2. Cannot support Hot Start
3. Maximum Number of Instruction Packet: 8 (Check $RMI_EXEC$BUF_SIZE) (if the buffer overload, return error code RMIT-028)
4. Cannot support motion shorter than 40 ms; otherwise, robot will slow down (Check $RMI_CFG$MIN_ITP_TIME)
Author: Hsien-Chung Lin
Created Date: 2020/01/16
Modified Date: 2021/02/24 (Add 1. motion option, 2. circular motion, 3. error handling and display)
Last modified: Yu Zhao, 2023/09/28, add rmCall function to call TP program
Last modified: Yu Zhao, 2023/10/03, exit()->sys.exit() to enable exiting in jupyter
Last modified: Yu Zhao, 2023/10/04, exit()->return to avoid hard exit
'''

import socket, json, time, sys, math
import numpy as np
from collections import defaultdict, OrderedDict
from queue import Queue
from threading import Thread, Event # Try to do multi-thread, current is unused package
import sys

class pyRMI:
    def __init__(self, ip = '127.0.0.1'):
        self.ip = ip            # target ip address (ROBOGUIDE: 127.0.0.1)
        self.port = 16001       # target port number (default:16001)
        self.sock = None        # TCP/IP socket
        self.connect = False    # socket connection
        self.ErrorID = 0        # Error ID from Robot Controller
        self.ServoReady = 0     # 1: Ready for Motion; otherwise in error condition
        self.TPmode = 0         # Teach Pendant Disable (0) or Enable (1)
        self.RMIMotionStatus = 0# Remote Motion Interface running (1), not running (0)
        self.ProgramStatus = 0  # RMI_Move TP program's current status. If 1 the RMI_Move is aborted 
        self.SingleStepMode = 0 # 1: In Single Step Mode
        self.NoUTool = 0        # Number of User Tools available 
        self.NoUFrame = 0       # Number of User Frames available
        self.moMessage = ''     # motion Message
        self.seqID = 1          # Sequence Number
        self.recvSeqID = 0      # Sequence Number received from Robot
        self.sentSeqID = 0      # Sequence Number sent to Robot
        self.termType = '"FINE"'  # Term Type: FINE, CNT, CR (required Advanced Constant Path)
        self.termVal = 100      # Term Value: 1-100 (FINE ignores Term Value)
        self.spdType = '"Percent"'# Speed Type: "Percent", "mmSec", "InchMin", "Time(msec)"
        self.spd = 100          # Speed Value [Percent:%; mmSec: mm/s; InchMin: inch/min; Time: msec]
        self.cfg = {"UToolNumber":1,"UFrameNumber":1,
                    "Front":1,"Up":1,"Left":0,"Flip":0,
                    "Turn4":0,"Turn5":0,"Turn6":0}     # robot configuration (for Cartesian space motion)
        self.cartPosForm = ["X","Y","Z","W","P","R","Ext1","Ext2","Ext3"]
        self.jntPosForm = ["J1","J2","J3","J4","J5","J6","J7","J8","J9"]  
        self.verbose = False    # opton to print message
        self.buffer = b''       # buffer for response
        self.curSpdOvd = 100    # default Speed Override
        self.isAutoSend = True  # option for auto sending message
        self.motionBuffer = Queue() # Motion Queue control in both RC side and PC side 
        self.rcBufferSize = 4 # Robot Controller Buffer Size (See Sysvar: $RMI_CFG.$EXEC_SIZE in the robot controller) default 4
        self.recvBuffer = defaultdict(list) # Recv Queue manage the response message sent from RC
        self.isUseRecvBuffer = False
        self.rmiDone = False
        self.closeEvent = Event()
        self.timeout = 5.0 # set 5s for sock timeout
        

    def rmInitSocket(self):
        '''
        Initialize TCP/IP socket for robot communication
        '''
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print('Try to connect to ip: %s' %self.ip + ', port: %s' %str(self.port))
            self.sock.connect((self.ip, self.port))
            self.sock.settimeout(self.timeout)
            print('Remote Motion Interface Socket Established')
            self.connect = True
        except Exception as e: 
            print(e)
            self.sock.close()
            # sys.exit()
            return
    
    def rmConnect(self):
        self.sock.send(b'{"Communication":"FRC_Connect"}\r\n')
        response = self.rmGetConfirmation('FRC_Connect')
        if self.ErrorID == 0: 
            print('Remote Motion Interface Connected')
            self.port = response["PortNumber"] 
        else: 
            print("Fail to connect, close socket, and exit the program")
            self.sock.close()
            # sys.exit()
            self.ServoReady=0
            self.rmiDone=True
            return
    
    def rmDisconnect(self, forceClose = 0):
        self.sock.send(b'{"Communication":"FRC_Disconnect"}\r\n')
        self.rmiDone = True
        if forceClose == 0: 
            self.rmGetConfirmation('FRC_Disconnect')
            if self.ErrorID == 0: 
                self.sock.close()
                print('Remote Motion Interface Disconnected')
        else: # If it is force close, direct close socket without asking feedback to avoid infity loop
            self.sock.close()
            print('[Error] Force close Remote Motion Interface')
        
    def rmInitialize(self):
        self.sock.send(b'{"Command":"FRC_Initialize"}\r\n')
        self.rmGetConfirmation('FRC_Initialize')
        if self.ErrorID == 0: print('Remote Motion Initialized')
    
    def rmAbort(self, forceClose=0):
        self.sock.send(b'{"Command":"FRC_Abort"}\r\n')
        if forceClose == 0:
            self.rmGetConfirmation('FRC_Abort')
            if self.ErrorID == 0: print('Remote Motion Command Aborted')
        else: # If it is force close, direct close socket without asking feedback to avoid infity loop
            print("Force Abort the RMI Program")        

    def rmPause(self):
        self.sock.send(b'{"Command":"FRC_Pause"}\r\n')
        self.rmGetConfirmation('FRC_Pause')
        if self.ErrorID == 0: print('Remote Motion Command Paused')

    def rmContinue(self):
        self.sock.send(b'{"Command":"FRC_Continue"}\r\n')
        self.rmGetConfirmation('FRC_Continue')
        if self.ErrorID == 0: print('Remote Motion Command Continued')

    def rmReadError(self):
        self.sock.send(b'{"Command":"FRC_ReadError"}\r\n')
        # response = self.rmGetResponse() # Avoid using GetResponse which might trap in dead loop
        try:
            response = self.sock.recv(1024)
        except Exception as e: 
            print(e)
            self.rmDisconnect()
        if self.verbose: print('received message: %s' % response)
        
        response = response.decode('utf-8')
        response = self.padZero(response,' .0')
        response = self.padZero(response,'-.0')
        if len(response) >0:
            response = json.loads(response)
        else:
            print("[Error] recive empty packet from robot, check robot status")
            # sys.exit(forceClose=1)
            self.ServoReady=0
            self.rmiDone=True
            return
        self.ErrorID = response["ErrorID"]
        
        if 'FRC_ReadError' in response.values():
            errorData = response["ErrorData"]
            print('Error Data: %s'  % errorData)
            # self.closeRMI(forceClose=1)
            # sys.exit()
            self.ServoReady=0
            self.rmiDone=True
            return
        else: 
            print('[Error] Unknown Message, Get RMI response below:')
            for key, val in response.items(): print(str(key) + ' : ' + str(val))
            # self.closeRMI(forceClose=1)
            # sys.exit()
            self.ServoReady=0
            self.rmiDone=True
            return


    def rmSetUFrameUToolNo(self,UFrameNo, UToolNo):
        cmdMsg = '"Command":"FRC_SetUFrameUTool",'
        frameNoMsg = '"UFrameNumber":' + str(UFrameNo) + ','
        toolNoMsg = '"UToolNumber":' + str(UToolNo) 
        message = '{' + cmdMsg + frameNoMsg + toolNoMsg + '}\r\n'
        self.rmSendMessage(message)
        self.cfg["UToolNumber"] = UToolNo
        self.cfg["UFrameNumber"] = UFrameNo
        self.rmGetConfirmation('FRC_SetUFrameUTool')

    def rmGetStatus(self):
        self.sock.send(b'{"Command":"FRC_GetStatus"}\r\n')
        response = self.rmGetConfirmation('FRC_GetStatus')
        self.ErrorID = response["ErrorID"]
        self.ServoReady = response["ServoReady"]
        self.TPmode = response["TPMode"]
        self.RMIMotionStatus = response["RMIMotionStatus"]
        self.ProgramStatus = response["ProgramStatus"] 
        self.SingleStepMode = response["SingleStepMode"]
        self.NoUTool = response["NumberUTool"] 
        self.NoUFrame = response["NumberUFrame"]

    def rmReadDIN(self,portNo):
        cmdMsg = '"Command":"FRC_ReadDIN",'
        portNoMsg = '"PortNumber":' + str(portNo) 
        message = '{'+cmdMsg + portNoMsg + '}\r\n'
        self.rmSendMessage(message)
        response = self.rmGetConfirmation('FRC_ReadDIN')    
        portNo = response["PortNumber"]
        portVal = response["PortValue"]
        return portVal, portNo

    def rmWriteDOUT(self,portNo,portVal):
        cmdMsg = '"Command":"FRC_WriteDOUT",'
        portNoMsg = '"PortNumber":' + str(portNo) + ','
        portValMsg = '"PortValue":' + str(portVal)
        message = '{'+cmdMsg + portNoMsg + portValMsg +'}\r\n'
        self.rmSendMessage(message)
        self.rmGetConfirmation('FRC_WriteDOUT')

    def rmGetCartPos(self):
        self.sock.send(b'{"Command":"FRC_ReadCartesianPosition"}\r\n')
        response = self.rmGetConfirmation('FRC_ReadCartesianPosition')
        CartPos = response["Position"]
        Config = response["Configuration"]
        TimeTag = response["TimeTag"]
        return CartPos, Config, TimeTag, response
    
    def rmGetJntPos(self):
        self.sock.send(b'{"Command":"FRC_ReadJointAngles"}\r\n')
        response = self.rmGetConfirmation('FRC_ReadJointAngles')
        JntPos = response.get("JointAngle")
        if JntPos == None: JntPos = response.get("JointAngles") # Handle Older Version Issue 
        TimeTag = response["TimeTag"]
        return JntPos, TimeTag
    
    def rmSetSpdOvd(self,val):
        cmdMsg = '"Command":"FRC_SetOverRide",'
        valMsg = '"Value":' +str(val)
        message = '{' + cmdMsg + valMsg + '}\r\n'
        self.rmSendMessage(message)
        self.rmGetConfirmation('FRC_SetOverRide')

    
    def rmGetUFrameUTool(self):
        self.sock.send(b'{"Command":"FRC_GetUFrameUTool"}\r\n')
        response = self.rmGetConfirmation('FRC_GetUFrameUTool')
        UFrameNo = response["UFrameNumber"]
        UToolNo = response["UToolNumber"]
        return UFrameNo, UToolNo
    
    def rmGetPR(self,prNo):
        cmdMsg = '"Command":"FRC_ReadPositionRegister",'
        prMsg = '"RegisterNumber":' +str(int(prNo)) + ','
        message = '{' + cmdMsg + prMsg + '}\r\n'
        self.rmSendMessage(message)
        response = self.rmGetConfirmation('FRC_ReadPositionRegister')
        prNo = response["RegisterNumber"]
        Config = response["Configuration"]
        CartPos = response["Position"]
        return CartPos, Config, prNo

    def rmWritePR(self, prNo, cartPos, config = None):
        if config == None: config = self.cfg
        cmdMsg = '"Command":"FRC_WritePositionRegister",'
        prMsg = '"RegisterNumber":' +str(int(prNo)) + ','
        cfgMsg = self.rmWriteCfgMsg(config)
        posMsg = self.rmWritePosMsg(cartPos,self.cartPosForm)
        message = '{' + cmdMsg + prMsg + cfgMsg + posMsg +'}\r\n'
        self.rmSendMessage(message)
        self.rmGetConfirmation('FRC_WritePositionRegister')

    def rmReset(self):
        self.sock.send(b'{"Command":"FRC_Reset"}\r\n')
        self.rmGetConfirmation('FRC_Reset')
    
    def rmGetTcpSpd(self):
        self.sock.send(b'{"Command":"FRC_ReadTCPSpeed"}\r\n')
        response = self.rmGetConfirmation('FRC_ReadTCPSpeed')
        tcpSpd = response["Speed"]
        return tcpSpd

    def rmWaitDI(self,portNo,portVal):
        cmdMsg = '"Instruction":"FRC_WaitForDin",'  # Newer Version: WaitDin; Older version: WaitForDin
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID += 1 
        portNoMsg = '"PortNumber":' + str(portNo) +','
        portValMsg = '"PortValue"' + str(portVal)
        message = '{' + cmdMsg +seqMsg + portNoMsg + portValMsg +'}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)
        self.rmGetConfirmation('FRC_WaitForDin')

    def rmSetUFrame(self,UFrameNo):
        # Error in Old Version (V9.3044), show error code: SYST-040
        cmdMsg = '"Instruction":"FRC_SetUFrame",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID += 1
        frameNoMsg = '"FrameNumber":' + str(UFrameNo)
        message = '{' + cmdMsg + seqMsg + frameNoMsg +'}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)
        self.rmGetConfirmation('FRC_SetUFrame')
        self.cfg["UFrameNumber"] = UFrameNo

    def rmSetUTool(self,UToolNo):
        # Error in Old Version (V9.3044), show error code: SYST-040
        cmdMsg = '"Instruction":"FRC_SetUTool",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID += 1
        toolNoMsg = '"ToolNumber":' + str(UToolNo)
        message = '{' + cmdMsg + seqMsg + toolNoMsg +'}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)
        self.rmGetConfirmation('FRC_SetUTool')        
        self.cfg["UToolNumber"] = UToolNo

    def rmWaitTime(self,time):
        cmdMsg = '"Instruction":"FRC_WaitTime",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        timeMsg = '"Time":' + str(time)
        message = '{' + cmdMsg + seqMsg + timeMsg +'}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)

    def rmSetPayload(self,payload):
        cmdMsg = '"Instruction":"FRC_SetPayLoad",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        toolNoMsg = '"ScheduleNumber":' + str(payload)
        message = '{' + cmdMsg + seqMsg + toolNoMsg +'}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)
        self.rmGetConfirmation('FRC_SetPayLoad')

    def rmLinearMotion(self, pos, config = None, spdType = "mmSec", spd = 800, termType = None, termVal = None, **motionOptions):
        if config == None: config = self.cfg
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        cmdMsg = '"Instruction":"FRC_LinearMotion",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        cfgMsg = self.rmWriteCfgMsg(config)
        posMsg = self.rmWritePosMsg(pos, self.cartPosForm) + ','
        spdMsg = '"SpeedType":' + spdType + ',' + '"Speed":' + str(spd) + ','
        termMsg = '"TermType":' + termType + ',' + '"TermValue":' + str(termVal)
        optMsg = '' if len(motionOptions) == 0 else self.rmWriteOptMsg(motionOptions) 
        message = '{' + cmdMsg + seqMsg + cfgMsg + posMsg + spdMsg + termMsg + optMsg + '}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)

    def rmLinearRelative(self, pos, config = None, spdType = "mmSec", spd = 1500, termType = None, termVal = None, **motionOptions):
        if config == None: config = self.cfg
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        cmdMsg = '"Instruction":"FRC_LinearRelative",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        cfgMsg = self.rmWriteCfgMsg(config)
        posMsg = self.rmWritePosMsg(pos, self.cartPosForm) + ','
        spdMsg = '"SpeedType":' + spdType + ',' + '"Speed":' + str(spd) + ','
        termMsg = '"TermType":' + termType + ',' + '"TermValue":' + str(termVal)
        optMsg = '' if len(motionOptions) == 0 else self.rmWriteOptMsg(motionOptions) 
        message = '{' + cmdMsg + seqMsg + cfgMsg + posMsg + spdMsg + termMsg + optMsg + '}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)

    def rmLinearMotionJRep(self, pos, spdType = None, spd =None, termType = None, termVal = None, **motionOptions):
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        cmdMsg = '"Instruction":"FRC_LinearMotionJRep",' 
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        posMsg = self.rmWritePosMsg(pos, self.jntPosForm) + ','
        spdMsg = '"SpeedType":' + spdType + ',' + '"Speed":' + str(spd) + ','
        termMsg = '"TermType":' + termType + ',' + '"TermValue":' + str(termVal)
        optMsg = '' if len(motionOptions) == 0 else self.rmWriteOptMsg(motionOptions)
        message = '{' + cmdMsg + seqMsg + posMsg + spdMsg + termMsg + optMsg + '}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)

    def rmLinearRelativeJRep(self, pos, spdType = None, spd =None, termType = None, termVal = None, **motionOptions):
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        cmdMsg = '"Instruction":"FRC_LinearRelativeJRep",' 
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        posMsg = self.rmWritePosMsg(pos, self.jntPosForm) + ','
        spdMsg = '"SpeedType":' + spdType + ',' + '"Speed":' + str(spd) + ','
        termMsg = '"TermType":' + termType + ',' + '"TermValue":' + str(termVal)
        optMsg = '' if len(motionOptions) == 0 else self.rmWriteOptMsg(motionOptions)
        message = '{' + cmdMsg + seqMsg + posMsg + spdMsg + termMsg + optMsg + '}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)

    def rmJointMotion(self, pos, config = None, spdType = None, spd =None, termType = None, termVal = None, **motionOptions):
        if config == None: config = self.cfg
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        cmdMsg = '"Instruction":"FRC_JointMotion",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        cfgMsg = self.rmWriteCfgMsg(config)
        posMsg = self.rmWritePosMsg(pos, self.cartPosForm) + ','
        spdMsg = '"SpeedType":' + spdType + ',' + '"Speed":' + str(spd) + ','
        termMsg = '"TermType":' + termType + ',' + '"TermValue":' + str(termVal)
        optMsg = '' if len(motionOptions) == 0 else self.rmWriteOptMsg(motionOptions) 
        message = '{' + cmdMsg + seqMsg + cfgMsg + posMsg + spdMsg + termMsg + optMsg + '}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)

    def rmJointRelative(self, pos, config = None, spdType = None, spd =None, termType = None, termVal = None, **motionOptions):
        if config == None: config = self.cfg
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        cmdMsg = '"Instruction":"FRC_JointRelative",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        cfgMsg = self.rmWriteCfgMsg(config)
        posMsg = self.rmWritePosMsg(pos, self.cartPosForm) + ','
        spdMsg = '"SpeedType":' + spdType + ',' + '"Speed":' + str(spd) + ','
        termMsg = '"TermType":' + termType + ',' + '"TermValue":' + str(termVal)
        optMsg = '' if len(motionOptions) == 0 else self.rmWriteOptMsg(motionOptions) 
        message = '{' + cmdMsg + seqMsg + cfgMsg + posMsg + spdMsg + termMsg + optMsg + '}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)

    def rmCircMotion(self, pos, viaPos, config = None, viaConfig = None, spdType = "mmsec", spd = 1500, termType = None, termVal = None, **motionOptions):
        if config == None: config = self.cfg
        if viaConfig == None: viaConfig = self.cfg
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        cmdMsg = '"Instruction":"FRC_CircularMotion",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        cfgMsg = self.rmWriteCfgMsg(config)
        posMsg = self.rmWritePosMsg(pos, self.cartPosForm) + ','
        viacfgMsg = self.rmWriteCfgMsg(viaConfig,1)
        viaposMsg = self.rmWritePosMsg(viaPos, self.cartPosForm,1) + ','
        spdMsg = '"SpeedType":' + spdType + ',' + '"Speed":' + str(spd) + ','
        termMsg = '"TermType":' + termType + ',' + '"TermValue":' + str(termVal)
        optMsg = '' if len(motionOptions) == 0 else self.rmWriteOptMsg(motionOptions) 
        message = '{' + cmdMsg + seqMsg + cfgMsg + posMsg + viacfgMsg + viaposMsg + spdMsg + termMsg + optMsg + '}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)        

    def rmCircRelative(self, pos, viaPos, config = None, viaConfig = None, spdType = "mmsec", spd = 1500, termType = None, termVal = None, **motionOptions):
        if config == None: config = self.cfg
        if viaConfig == None: viaConfig = self.cfg
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        cmdMsg = '"Instruction":"FRC_CircularRelative",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        cfgMsg = self.rmWriteCfgMsg(config)
        posMsg = self.rmWritePosMsg(pos, self.cartPosForm) + ','
        viacfgMsg = self.rmWriteCfgMsg(viaConfig,1)
        viaposMsg = self.rmWritePosMsg(viaPos, self.cartPosForm,1) + ','
        spdMsg = '"SpeedType":' + spdType + ',' + '"Speed":' + str(spd) + ','
        termMsg = '"TermType":' + termType + ',' + '"TermValue":' + str(termVal)
        optMsg = '' if len(motionOptions) == 0 else self.rmWriteOptMsg(motionOptions) 
        message = '{' + cmdMsg + seqMsg + cfgMsg + posMsg + viacfgMsg + viaposMsg + spdMsg + termMsg + optMsg + '}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)

    # Call TP program
    def rmCall(self, TPname: str):
        cmdMsg = '"Instruction":"FRC_Call",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        tppMsg = '"ProgramName":' + TPname
        message = '{' + cmdMsg + seqMsg + tppMsg + '}\r\n'
        self.rmSendMessage(message,1)

    # confirm TP finish
    def confirmCall(self):
        response = self.rmGetConfirmation('FRC_Call')
        if self.ErrorID != 0: 
            print('error message: %s' % response)

    def rmJointMotionJRep(self, pos, spdType = None, spd =None, termType = None, termVal = None, **motionOptions):
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        cmdMsg = '"Instruction":"FRC_JointMotionJRep",' 
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        posMsg = self.rmWritePosMsg(pos, self.jntPosForm) + ','
        spdMsg = '"SpeedType":' + spdType + ',' + '"Speed":' + str(spd) + ','
        termMsg = '"TermType":' + termType + ',' + '"TermValue":' + str(termVal)
        optMsg = '' if len(motionOptions) == 0 else self.rmWriteOptMsg(motionOptions)
        message = '{' + cmdMsg + seqMsg + posMsg + spdMsg + termMsg + optMsg + '}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)

    def rmJointRelativeJRep(self, pos, config = None, spdType = None, spd =None, termType = None, termVal = None, **motionOptions):
        if config == None: config = self.cfg
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        cmdMsg = '"Instruction":"FRC_JointRelativeJRep",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        posMsg = self.rmWritePosMsg(pos, self.jntPosForm) + ','
        spdMsg = '"SpeedType":' + spdType + ',' + '"Speed":' + str(spd) + ','
        termMsg = '"TermType":' + termType + ',' + '"TermValue":' + str(termVal)
        optMsg = '' if len(motionOptions) == 0 else self.rmWriteOptMsg(motionOptions) 
        message = '{' + cmdMsg + seqMsg + posMsg + spdMsg + termMsg + optMsg + '}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message,1)
        else:
            self.motionBuffer.put(message)

    def rmSMotion(self,pos,config = None, spdType = "mmSec", spd = 1500, termType = None, termVal = None, firstMove = None):
        # Not support in standard controllers
        if config == None: config = self.cfg
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        cmdMsg = '"Instruction":"FRC_SMotion",'
        seqMsg = '"SequenceID":' + str(self.seqID) + ','; self.seqID+=1
        cfgMsg = self.rmWriteCfgMsg(config)
        if firstMove == True:
            posMsg = self.rmWriteMultiPosMsg(pos, self.cartPosForm) + ','
            self.seqID +=2
        else:
            posMsg = self.rmWritePosMsg(pos, self.cartPosForm) + ','
        spdMsg = '"SpeedType":' + spdType + ',' + '"Speed":' + str(spd) + ','
        termMsg = '"TermType":' + termType + ',' + '"TermValue":' + str(termVal)
        message = '{' + cmdMsg + seqMsg + cfgMsg + posMsg + spdMsg + termMsg + '}\r\n'
        if self.isAutoSend == True:
            self.rmSendMessage(message)
        else:
            self.moMessage += message
        
    def rmMultiSMotion(self,pos,config = None, spdType = "mmSec", spd = 1500, termType = None, termVal = None):
        # Not support in standard controllers
        if config == None: config = self.cfg
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        pos = np.array(pos)
        numPt = np.shape(pos)[0]
        if numPt<3: print("Position less than 3, can't generate spline"); return
        self.rmSMotion(pos[0:3], spd = spd, termType = "CNT",firstMove=True)
        for i in range(3,numPt):
            term = "FINE" if i==numPt-1 else "CNT"
            self.rmSMotion(pos[i],spd = spd, termType=term)

    def rmMultiJMotion(self,pos,config = None, spdType = "Percent", spd = 100, termType = None, termVal = None):
        if config == None: config = self.cfg
        if spdType == None: spdType = self.spdType
        if spd == None: spd = self.spd
        if termType == None: termType = self.termType
        if termVal == None: termVal = self.termVal
        pos = np.array(pos)
        numPt = np.shape(pos)[0]
        for i in range(numPt):
            term = "FINE" if i==numPt-1 else "CNT"
            self.rmJointMotion(pos[i],spd = spd, termType=term)

    def rmWriteOptMsg(self, motionOptions):
        msg = ''
        for key, val in motionOptions.items():
            msg += ',"' + str(key) + '":' + str(val) 
        return msg 

    def rmWriteCfgMsg(self,config, viaflag = 0):
        cfgMsg = '"Configuration":{'
        if viaflag == 1: cfgMsg = cfgMsg[:1] + 'Via' + cfgMsg[1:]
        cfgMsg += '"UToolNumber": ' + str(config["UToolNumber"]) + ','
        cfgMsg += '"UFrameNumber": ' + str(config["UFrameNumber"]) + ','
        cfgMsg += '"Front":' + str(config["Front"]) + ','
        cfgMsg += '"Up":' + str(config["Up"]) + ','
        cfgMsg += '"Left":' + str(config["Left"]) + ','
        cfgMsg += '"Flip":' + str(config["Flip"]) + ','
        cfgMsg += '"Turn4":' + str(config["Turn4"]) + ','
        cfgMsg += '"Turn5":' + str(config["Turn5"]) + ','
        cfgMsg += '"Turn6":' + str(config["Turn6"]) + '},'
        return cfgMsg

    def rmWritePosMsg(self, pos, form, viaflag = 0):
        if form[0] == "X": posMsg = '"Position":{'
        elif form[0] =="J1": posMsg = '"JointAngle":{'  # V93044 is "JointAngles" and newer version is "JointAngle" (no s)
        if viaflag == 1: posMsg = posMsg[:1] + 'Via' + posMsg[1:]
        for i in range(np.size(pos)):
            posMsg += '"'+ form[i] +'":'+ str(float(pos[i]))
            posMsg += '}' if i == np.size(pos)-1 else ','
        return posMsg

    def rmWriteMultiPosMsg(self, pos, form):
        numPos,dim = np.shape(pos)
        posMsg = '"Position":{'
        for j in range(numPos):
            for i in range(dim):
                frm = form[i] +str(j+1) if j!=0 else form[i]
                posMsg += '"'+ frm +'":'+ str(float(pos[j,i]))
                posMsg += '}' if i+j == numPos+dim-2 else ','
        return posMsg

    def rmSendMessage(self,message, updateSeq = 0):
        if self.verbose: print('send message: %s' % message)
        message = message.encode('utf-8')
        self.sock.send(message)
        if updateSeq != 0: self.sentSeqID += updateSeq

    def rmRecievePacket(self):
        delim = b'\r\n'
        buffer = b''
        response=b''
        try:
            sock_start_time = time.time()
            response = self.sock.recv(1024)
            sock_end_time = time.time()
            print('sock time', sock_end_time - sock_start_time)
        except Exception as e:
            print(e)
            self.closeRMI(forceClose=1)
        if self.verbose or True: print('received message: %s' % response)
        buffer += response

        while delim in buffer:
            # split buffer, get single response message
            i = buffer.find(delim)
            response = buffer[:i]
            i += len(delim)
            buffer = buffer[i:]

            # parse response json
            response = response.decode('utf-8')
            response = self.padZero(response,' .0')
            response = self.padZero(response,'-.0')
            if len(response) >0:
                response = json.loads(response)
            else:
                print("[Error] recive empty packet from robot, check robot connection")
                # self.closeRMI(forceClose=1)
                # sys.exit()
                self.ServoReady=0
                self.rmiDone=True
                return

            # Handling some Error Conditions
            if 'FRC_SystemFault' in response.values():
                print('[ERROR] Robot System Fault, Terminate the RMI')
                # self.closeRMI(forceClose=1)
                # sys.exit()
                self.ServoReady=0
                self.rmiDone=True
                return
            if 'Unknown' in response.values():
                print('[Error] RMI cannot interpret the string')
                # self.closeRMI(forceClose=1)
                # sys.exit()
                self.ServoReady=0
                self.rmiDone=True
                return()
            if 'ErrorID' in response:
                if response["ErrorID"] != 0:
                    # self.rmReadError()
                    pass # readerror may lead to wrong errorID

            # Update recvSeqID
            if 'SequenceID' in response: self.recvSeqID = response["SequenceID"]
            
            # Retrieve Key and Store to RecvBuffer
            '''
            Multi-thread for recvpacket can independent process the receive packet, 
            it is not necessary use the following code to process confirmation. 
            Comment out for now and consider to how use recvbuffer in the future. 
            '''
            # if 'Communication' in response: key = response["Communication"]
            # if 'Command' in response: key = response["Command"]
            # if 'Instruction' in response: key = response["Instruction"]
            # self.recvBuffer[key].append(response)
            
    def runRecevPack(self):
        while not self.rmiDone:
            try:
                self.rmRecievePacket()
            except KeyboardInterrupt:
                break

    def settimeout(self, timeout):
        self.sock.settimeout(timeout)

    def rmGetResponse(self, timeout = 5.0):
        delim = b'\r\n'
        response = b''
        buffer = self.buffer
        if not delim in buffer:
            # self.sock.settimeout(timeout)
            try:
                sock_start_time = time.time()
                response = self.sock.recv(1024)
                sock_end_time = time.time()
                print('sock time', sock_end_time - sock_start_time)
            except TimeoutError:
                return response
            except Exception as e: 
                print(e)
                self.rmDisconnect()
                return response
            if self.verbose or True: print('received message: %s' % response)
            # self.sock.settimeout(None)
            buffer += response
        # split buffer, get single response message
        i = buffer.find(delim)
        response = buffer[:i]
        i += len(delim)
        self.buffer = buffer[i:]

        # parse response json
        response = response.decode('utf-8')
        response = self.padZero(response,' .0')
        response = self.padZero(response,'-.0')
        if len(response) >0:
            response = json.loads(response)
        else:
            print("[Error] recive empty packet from robot, check robot connection")
            # self.closeRMI(forceClose=1)
            # sys.exit()
            self.ServoReady=0
            self.rmiDone=True
            return response

        # Handling some Error Conditions
        if 'FRC_SystemFault' in response.values():
            print('[ERROR] Robot System Fault, Terminate the RMI')
            # self.closeRMI(forceClose=1)
            # sys.exit()
            self.ServoReady=0
            self.rmiDone=True
            return response

        if 'Unknown' in response.values():
            print('[Error] RMI cannot interpret the string')
            # self.closeRMI(forceClose=1)
            # sys.exit()
            self.ServoReady=0
            self.rmiDone=True
            return response

        if 'SequenceID' in response: self.recvSeqID = response["SequenceID"]
        # if response["ErrorID"] != 0 and 'FRC_ReadError' not in response.values():    
        #     self.rmReadError() # Sometimes the Error ID only returned in the command return packet and can't be read by ReadError    
        return response

    def padZero(self,msg,targetStr):
        ###  fix the rmi numerical format issue in " .0" and "-.0"
        while msg.find(targetStr)>=0:
            indx = msg.find(targetStr) +1 # Padding zeros at the second indx
            msg = msg[:indx]+'0'+msg[indx:]
        return msg

    def rmGetErrorID(self):
        response = self.rmGetResponse()
        errorID = response["ErrorID"]
        return errorID

    def rmGetConfirmation(self, keyword):
        if not self.isUseRecvBuffer:
            response = self.rmGetResponse()
            cnt=0
            maxRetry=3-1
            while cnt<maxRetry and keyword not in response.values():
                response = self.rmGetResponse()
                cnt+=1
        else:
            '''
            Multi-thread for recvpacket can independent process the receive packet, 
            it is not necessary use the following code to process confirmation. 
            Comment out for now and consider to how use recvbuffer in the future. 
            '''
            # while len(self.recvBuffer[keyword]) == 0 and not self.closeEvent.set(): 
            #     # print("Wait for recvBuffer store confirmation")
            #     time.sleep(0.001)
            # if len(self.recvBuffer[keyword]) == 0: self.rmRecievePacket()
            # response = self.recvBuffer[keyword].pop()
            return ''

        if "ErrorID" not in response: print("Error in Comfirmation Packet")
        self.ErrorID = response["ErrorID"]
        if self.ErrorID > 0: 
            print('Fail to execute %s, ' %keyword + 'ErrorID: %s' %str(self.ErrorID))
            # self.rmReadError()
        return response
    
    def printRMIinfo(self): 
        info = vars(self)
        for key, val in info.items():
            print(key,val)
        del info
    
    def setIP(self,ip): self.ip = ip

    def setVerbose(self,val): self.verbose = val

    def isLastSentMotionNotDone(self): return self.recvSeqID < self.seqID-1

    def isLastLastSentMotionNotDone(self): return self.recvSeqID < self.seqID-2

    def isRCBufferFull(self): return (self.sentSeqID-self.recvSeqID) >= self.rcBufferSize

    def initializRMI(self):
        # Initialize the Connection with Robot Controller
        self.rmInitSocket()
        self.rmConnect()
        self.rmInitSocket()
        if self.verbose: print('Initialized the client socket connecting to ip: %s ' % self.ip + 'port: %s' % str(self.port))
        
        # Check the System Status, exit if the system is not ready
        self.rmGetStatus()
        if self.ServoReady != 1: 
            print( "[ERROR] SRVO not ready")
            self.rmDisconnect()
            # sys.exit()
            self.ServoReady=0
            self.rmiDone=True
            return # added
        
        if self.TPmode != 0:
            print( "[ERROR] Teach Pedant not in aoto mode")
            self.rmDisconnect()
            # sys.exit()
            self.ServoReady=0
            self.rmiDone=True
            return # added

        # Initialize the RMI socket (like press cycle start button)
        self.rmInitialize()
    
    def closeRMI(self, forceClose = 0):
        self.rmiDone=True
        self.rmAbort(forceClose)
        self.rmDisconnect(forceClose)
        

def main():
    # ip = '127.0.0.2'
    ip = '192.168.1.11'
    rmi = pyRMI(ip)
    rmi.setVerbose(True)

    # Initialize the RMI Socket
    rmi.initializRMI()
    # p1 = np.array([0, 0, 0, 0, -90, 0])
    # p2 = np.array([40, 0, 20, 0, -90, 0])
    # p3 = np.array([-40, 0, 10, 0, -90, 0])
    p1 = np.array([790, 0, 700, -180, 0, 0])
    p2 = np.array([400, 300, 600, -180, 0, 0])
    p3 = np.array([400, -300, 600, -180, 0, 0])
    # c1 = np.array([790,100, 700, -180, 0, 0])
    # c2 = np.array([890,0, 700, -180, 0, 0])
    # c3 = np.array([790,-100,700,-180,0,0])
    # c4 = np.array([690,0,700,-180,0,0])
    # p1 = np.array([1200, 0, 0, -180, 0, 0])
    # p2 = np.array([600, -600, 0, -180, 0, 0])
    # p3 = np.array([0, -1200, 0, -180, 0, 0])
    
    # Set up the User Frame Number and User Tool Nomber together
    UFrameNo = 2; UToolNo = 2
    rmi.rmSetUFrameUToolNo(UFrameNo,UToolNo) # equivalent to rmi.rmSetUTool(1) + rmi.rmSetUFrame(1)
    
    # Setup UFrame and UTool individually  (Not works for V9.3044 but works for newer versions)
    # rmi.rmSetUFrame(UFrameNo)
    # rmi.rmSetUTool(UToolNo)

    # Get UFrame and UTool Number
    UFrame_rob, UTool_rob = rmi.rmGetUFrameUTool()
    print('UFrame at Robot: %s' % str(UFrame_rob) + ', UTool at Robot: %s' % str(UTool_rob) )

    # Set Up and Get Position Register
    prNo = 1
    prPos = np.array([0,0,10,0,0,0])
    rmi.rmWritePR(prNo, prPos)
    prPos_rob, prCfg_rob, prNo_rob  = rmi.rmGetPR(prNo)
    posPrintStr = "".join(str(key) + ':' + str(value) + ', ' for key, value in prPos_rob.items())
    cfgPrintStr = "".join(str(key) + ':' + str(value) + ', ' for key, value in prCfg_rob.items())
    print("PR[%s] - POS: " % prNo_rob + '[' + posPrintStr + ']')
    print("PR[%s] - CFG: " % prNo_rob + '[' + cfgPrintStr + ']')

    # Setup Speed Override 
    rmi.rmSetSpdOvd(50)

    # Setup Payload
    rmi.rmSetPayload(1)

    # Read DIN PortNo
    DINPortNo_req = 1
    DINPortVal, DINPortNo = rmi.rmReadDIN(DINPortNo_req)
    print("DI[%s]: " %str(DINPortNo) + '%s' %str(DINPortVal))

    # Write DOUT
    DOUTPortNo_write = 2
    DOUTPortVal_write = "ON"
    rmi.rmWriteDOUT(DOUTPortNo_write,DOUTPortVal_write)

    # Read Cart Pos
    cartPos, cfg, robtime = rmi.rmGetCartPos()
    print("Cart Pos: [" + "".join(str(key) + ' : ' + str(value) + ' ' for key, value in cartPos.items()) + "]")
    print("Cfg: [" + "".join(str(key) + ' : ' + str(value) + ' ' for key, value in cfg.items()) + "]")
    print("Time: " +str(robtime))

    # Read Joint Pos
    jntPos, robtime = rmi.rmGetJntPos()
    print("Jnt Pos: [" + "".join(str(key) + ' : ' + str(value) + ' ' for key, value in jntPos.items()) + "]")
    print("Time: " +str(robtime))

    # # Basic Linear & Joint Motion with Motion Option
    rmi.rmLinearMotion(p1)
    rmi.rmLinearMotion(p2,LCBType="DB", LCBValue=1000, PortType=1, PortNumber=3, PortValue="ON")
    # rmi.rmJointMotion(p1,ACC=80, OffsetPRNumber=prNo)
    rmi.rmLinearMotion(p3,WristJoint="ON")
    rmi.rmJointMotion(p1, LCBType="TA", LCBValue=100, PortType=1, PortNumber=3, PortValue="OFF")

    # # Circular Motion
    # rmi.rmLinearMotion(c1)
    # rmi.rmCircMotion(c2, c3, spd=100, termType="CNT")
    # rmi.rmCircMotion(c4, c1, spd=100)
    
    while rmi.isLastSentMotionNotDone(): rmi.rmGetResponse()

    # # Use Motion Buffer to store/send command
    rmi.isAutoSend = False
    rmi.rmLinearMotion(p2, termType="CNT")
    rmi.rmJointMotion(p3, termType="CNT")
    rmi.rmJointMotion(p1)
    rmi.rmJointMotionJRep([0,0,0,0,-90,0])
    while not rmi.motionBuffer.empty():
        rmi.rmSendMessage(rmi.motionBuffer.get())
        
    while rmi.isLastSentMotionNotDone(): rmi.rmGetResponse()

    rmi.closeRMI()

def threading_test():
    ip = '192.168.1.101'
    rmi = pyRMI(ip)
    rmi.setVerbose(True)

    # Initialize the RMI Socket
    rmi.initializRMI()
    
    rmi.isUseRecvBuffer = True

    reciver = Thread(name='RMI_Reciever',target=rmi.runRecevPack) 
    reciver.start()

    # Set up the User Frame Number and User Tool Nomber together
    UFrameNo = 2; UToolNo = 2
    rmi.rmSetUFrameUToolNo(UFrameNo,UToolNo) # equivalent to rmi.rmSetUTool(1) + rmi.rmSetUFrame(1)

    p1 = np.array([790, 0, 700, -180, 0, 0])
    p2 = np.array([400, 300, 600, -180, 0, 0])
    p3 = np.array([400, -300, 600, -180, 0, 0])
    # # Basic Linear & Joint Motion with Motion Option
    rmi.rmLinearMotion(p1)
    rmi.rmLinearMotion(p2,LCBType="DB", LCBValue=1000, PortType=1, PortNumber=3, PortValue="ON")
    rmi.rmLinearMotion(p3,WristJoint="ON")
    rmi.rmJointMotion(p1, LCBType="TA", LCBValue=100, PortType=1, PortNumber=3, PortValue="OFF")
    rmi.rmJointMotionJRep([0,0,0,0,-90,0])

    while rmi.isLastSentMotionNotDone():
        cartPos, _, robtime = rmi.rmGetCartPos()
        print('[' +str(robtime) + ']' + "Cart Pos: [" + "".join(str(key) + ' : ' + str(value) + ' ' for key, value in cartPos.items()) + "]")
    rmi.rmiDone = True
    reciver.join()
    rmi.closeRMI()


if __name__ == "__main__":
    main()


    




