from pymycobot import MyCobot280RDKX5
import time

mc=MyCobot280RDKX5("/dev/ttyS1")
mc.send_angles([0,0,0,0,0,0],100)
time.sleep(2)