import numpy as np
import socket, struct

# HSPO server from Hsien-Chung Lin
class pyHSPOServer():
    def __init__(self, port = 51002, timeout=1.0):
        #Socket Setup 
        self.ip = ''        # Server accept ip
        self.port = port    # high speed position output port (default = 60015)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)    # UDP socket sever
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.clisock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cliAddressBook = []
        self.debug = False
        self.prevPositionJnt = []      # Store Joint Position Value to calculate joint velocity
        self.prevRecvTimeJnt = None    # Store previous recive packet time stamp to calculate joint velocity
        self.prevPositionTcp = []      # Store TCP Position Value to calculate Tcp velocity
        self.prevRecvTimeTcp = None    # Store previous recive packet time stamp to calculate TCP velocity
        self.rostick2sec = 2/1000      # Convert Robot Controller rostick (2ms) to sec
        self.numVar = 6
        self.VarPktFormat = ">3L1H" + str(self.numVar) + "f" 
        self.FanucJntFormat = True  # Use FANUC Format (J2 + J3)
        self.JntPosDegree = True
        # self.timeout = timeout
        print("High Speed Position Output Server Initialized")
        
    def hspoConnect(self):
        try:
            self.sock.bind((self.ip, self.port))
            # self.sock.settimeout(self.timeout)
            print("High Speed Position Output Server Connect to Port %s" %str(self.port))
        except Exception as e:
            print("[ERROR] Fail to connect to robot client")
            print(e)
            exit(-1)

    def addClientAddress(self, clientIP, clientPort):
        self.cliAddressBook.append( (clientIP, clientPort) )
        print("Add one client (ip: %s, " %clientIP + "port: %s)" %str(clientPort))

    def sendToClients(self, packet):
        for clientAddress in self.cliAddressBook:
            self.clisock.sendto(packet, clientAddress)

    def close(self):
        self.sock.close()
        self.clisock.close()
        print("Close HSPO Server Socket")
    
    def processPacket(self, data):
        dataSize = len(data)
        # hspoVer = struct.unpack_from('>L', data, 0)
        hspoVer, idx, clock, pktType = struct.unpack_from('>3L1H',data, 0)
        if self.debug:
            print("Version: ", hspoVer)
            print("Index: ", idx)
            print("Clock: ", clock)
            print("Type: ", pktType)
            
        if pktType == 1 or pktType == 2:
            ret = self.decodeTcpPosPacket(data)
        elif pktType == 4 or pktType == 8:
            ret = self.decodeJointPosPacket(data)
        elif pktType == 16:
            ret = self.decodeVarPacket(data)
        else:
            print("[ERROR]: Wrong Data Type")
            # exit(-1)
            ret = {}
        
        return ret
        

    def decodeTcpPosPacket(self, data):
        value = struct.unpack('>3L2H6f2L',data) # Cartesian Position Packet
        ret = {}
        ret["version"] = value[0]
        ret["index"] = value[1]
        ret["clock"] = value[2]
        ret["type"] = value[3]
        ret["group"] = value[4]
        ret["position"] = np.array(value[5:11])  # Convert to Numpy Array for vector calculation
        ret["status"] = value[11]
        ret["io"] = value[12]
        
        # Calculate the velocity
        if self.prevRecvTimeTcp == None:
            ret["velocity"] = np.zeros(ret["position"].size)
        else:
            ret["velocity"] = (ret["position"] - self.prevPositionTcp) / (ret["clock"] - self.prevRecvTimeTcp) / self.rostick2sec

        self.prevRecvTimeTcp = ret["clock"]
        self.prevPositionTcp = ret["position"]

        # Convert to float list so that ROS Message could accept the data type (Comment out if you want to use numpy)
        # ret["position"] = ret["position"].tolist()
        # ret["velocity"] = ret["velocity"].tolist()

        return ret
    
    def decodeJointPosPacket(self, data):
        value = struct.unpack('>3L2H9f2L',data) # Joint Position Packet
        ret = {}
        ret["version"] = value[0]
        ret["index"] = value[1]
        ret["clock"] = value[2]
        ret["type"] = value[3]
        ret["group"] = value[4]
        ret["position"] = np.array(value[5:11])  # Convert to Numpy Array for vector calculation (Ignore J7-J9)
        ret["status"] = value[14]
        ret["io"] = value[15]

        if self.JntPosDegree:
            # convert position to degree
            ret['position']*=180./np.pi

        # Calculate the velocity
        if self.prevRecvTimeTcp == None:
            ret["velocity"] = np.zeros(ret["position"].size)
        else:
            ret["velocity"] = (ret["position"] - self.prevPositionTcp) / (ret["clock"] - self.prevRecvTimeTcp) / self.rostick2sec

        self.prevRecvTimeTcp = ret["clock"]
        self.prevPositionTcp = ret["position"]

        # Convert Joint Pos/Vel from FANUC convention to General Robotics convention
        if self.FanucJntFormat==False:
            ret["position"][2] += ret["position"][1]
            ret["velocity"][2] += ret["velocity"][1]

        # Convert to float list so that ROS Message could accept the data type (Comment out if you want to use numpy)
        # ret["position"] = ret["position"].tolist()
        # ret["velocity"] = ret["velocity"].tolist()

        return ret

    def decodeVarPacket(self, data):
        dataSize = len(data)
        # print("data size: ", dataSize)
        # value = struct.unpack(self.VarPktFormat, data)
        value = struct.unpack('>3L1H10f', data)
        ret = {}
        ret["version"] = value[0]
        ret["index"] = value[1]
        ret["clock"] = value[2]
        ret["type"] = value[3]
        ret["variable"] = np.array(value[4:4+self.numVar])

        # Convert to float list so that ROS Message could accept the data type (Comment out if you want to use numpy)
        # ret["variable"] = ret["variable"].tolist()

        return ret

