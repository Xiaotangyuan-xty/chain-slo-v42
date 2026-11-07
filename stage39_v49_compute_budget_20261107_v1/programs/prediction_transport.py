"""Explicit local predictor protocol for A/Astar/v48 alike; QRF unchanged."""
import pickle
import struct

PROTOCOL = 'length_framed_pickle_v1'
MAX_BYTES = 64 * 1024 * 1024


def recv_exact(connection, count):
    chunks = []
    while count:
        data = connection.recv(count)
        if not data:
            raise EOFError('Prediction frame ended before the declared length')
        chunks.append(data); count -= len(data)
    return b''.join(chunks)


def send_frame(connection, value, maximum=MAX_BYTES):
    data = pickle.dumps(value)
    if not 0 < len(data) <= maximum:
        raise ValueError('Prediction frame exceeds the explicit size budget')
    connection.sendall(struct.pack('!I', len(data)) + data)


def recv_frame(connection, maximum=MAX_BYTES):
    length, = struct.unpack('!I', recv_exact(connection, 4))
    if not 0 < length <= maximum:
        raise ValueError('Invalid prediction frame length')
    return pickle.loads(recv_exact(connection, length))


def handle(connection, predict, maximum=MAX_BYTES):
    request = recv_frame(connection, maximum)
    if request.get('protocol') != PROTOCOL or not isinstance(request.get('id'), str):
        raise ValueError('Prediction client/server protocol mismatch')
    prompts = request['prompt']
    if not isinstance(prompts, list) or len(prompts) != 1 or not isinstance(prompts[0], str):
        raise ValueError('Exactly one current prompt is required')
    value = predict(prompts)
    if len(value) != 1 or not isinstance(value[0], int) or value[0] < 1:
        raise ValueError('Invalid QRF length response')
    send_frame(connection, {'protocol':PROTOCOL, 'id':request['id'], 'output_len':value}, maximum)


def install_predictor(prediction, log=None):
    """Call in the predictor process before its original start_client()."""
    def handler(connection, model, vectorizer):
        try:
            with connection:
                handle(connection, lambda prompts:prediction.predict(vectorizer, model, prompts))
        except Exception as error:
            if log is not None:
                log('predictor_error', error_type=type(error).__name__, error=str(error))
            else:
                print('Framed prediction error: '+repr(error), flush=True)
    prediction.handle_request = handler
    return {'protocol':PROTOCOL, 'QRF_computation_changed':False, 'max_frame_bytes':MAX_BYTES}
