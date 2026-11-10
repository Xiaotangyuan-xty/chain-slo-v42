"""Keep bound listeners alive until the authorized child inherits each FD."""
import socket


def reserve(port):
    listener=socket.socket()
    try:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',port))
        listener.listen(128)
        return listener
    except BaseException:
        listener.close()
        raise
