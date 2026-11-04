import json,queue,threading,traceback
from pathlib import Path
class CallJSONLWriter:
    _STOP = object()

    def __init__(self, path):
        self.path = Path(path)
        self._queue = queue.Queue()
        self._ready = threading.Event()
        self._error = None
        self._exception = None
        self._finished = False
        self._thread = threading.Thread(target=self._run, name='client-calls-jsonl', daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            self._thread.join()
            self._raise_error()

    @property
    def error(self):
        return self._error

    def _capture_error(self, error):
        if self._error is None:
            self._error = {'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()}
            self._exception = error

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError(f'client_calls JSONL writer failed: {self._error['error']}') from self._exception

    def _run(self):
        try:
            handle = self.path.open('x')
        except BaseException as error:
            self._capture_error(error)
            self._ready.set()
            return
        self._ready.set()
        try:
            while True:
                value = self._queue.get()
                try:
                    if value is self._STOP:
                        break
                    if self._error is None:
                        handle.write(json.dumps(value, allow_nan=False, default=str) + '\n')
                        handle.flush()
                except BaseException as error:
                    self._capture_error(error)
                finally:
                    self._queue.task_done()
        finally:
            try:
                handle.close()
            except BaseException as error:
                self._capture_error(error)

    def record(self, value):
        if self._finished:
            raise RuntimeError('Cannot record after JSONL writer finish')
        self._queue.put(value)

    def drain(self):
        self._queue.join()
        self._raise_error()

    def finish(self):
        if not self._finished:
            self._finished = True
            self._queue.put(self._STOP)
            self._queue.join()
            self._thread.join()
        self._raise_error()
