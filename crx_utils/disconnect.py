from pyRemoteMotionInterface import pyRMI

rmi = pyRMI("192.168.1.101")
rmi.port = 16002
rmi.setVerbose(True)
rmi.rmInitSocket()
rmi.rmDisconnect()