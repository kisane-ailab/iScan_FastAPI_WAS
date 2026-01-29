import time
import zmq
import sys

if __name__ == "__main__":
    args = sys.argv

    port_num = "44444"
    cmd = 0x01

    if len(args) > 1:
        cmd = args[1]

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.bind("tcp://*:" + port_num)

    if cmd == 0x01:
        # 웜업 (0x01)
        message = "0x02" + "|" + "0x01" + "|" + "10" + "|" + "1" + "|" + "1" + "|" + "1" + "|" + "0x03"
        socket.send_string(message)
        message = socket.recv_string()
        print("WARM_UP : " + message + "\n")
    elif cmd == 0x02:
        message = "0x02" + "|" + "0xEE" + "|" + "0x03"
        socket.send_string(message)
        message = socket.recv_string()
        print("ALIVE : " + message + "\n")
    elif cmd == 0x03:
        message = \
            "0x02" + "|" + \
            "0x02" + "|" + \
            "3" + "|" + \
            "./Cam_2_Color.jpg" + "|" + \
            "./Cam_2_Color.jpg" + "|" + \
            "./Cam_1_Color.jpg" + "|" + \
            "True" + "|" + \
            "." + "|" + \
            "0x03"
        socket.send_string(message)
        message = socket.recv_string()
        print("INF : " + message + "\n")
    elif cmd == 0xFF:
        message = "0x02" + "|" + "0xFF" + "|" + "0x03"
        socket.send_string(message)
        message = socket.recv_string()
        print("QUIT: " + message + "\n")