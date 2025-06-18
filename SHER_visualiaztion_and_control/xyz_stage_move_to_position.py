from pipython import GCSDevice, pitools

# Q motion controller setup
CONTROLLERNAME = 'E-873'

with GCSDevice(CONTROLLERNAME) as pidevice:
        pidevice.ConnectTCPIP(ipaddress='192.168.1.220')
        pitools.startup(pidevice, stages='Q-522.130', refmodes='FNL')
        pidevice.MOV({'1': 0, '2': 0, '3': -5.5}) 