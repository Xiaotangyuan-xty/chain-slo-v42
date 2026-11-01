"""Process-local prediction socket routing; computation and payload are unchanged."""
import socket

def isolate_prediction_port(port, role):
    if role not in {'server','predictor'} or not isinstance(port,int) or not 1024<=port<=65535:
        raise ValueError('Invalid local predictor endpoint')
    original=socket.socket
    def route(address):
        return (address[0],port) if address in [('localhost',65433),('127.0.0.1',65433)] else address
    class IsolatedSocket(original):
        def connect(self,address):
            return super().connect(route(address) if role=='server' else address)
        def bind(self,address):
            return super().bind(route(address) if role=='predictor' else address)
    socket.socket=IsolatedSocket
    return {'role':role,'prediction_port':port,'payload_or_computation_changed':False}
