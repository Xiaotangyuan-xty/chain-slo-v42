"""Behavioral checks for lease packing, compound cache refresh and port isolation."""
import socket
import unittest
from chain_admission import select,blocks
from chain_forecast import StageForecaster
from transport_isolation import isolate_prediction_port

def member(name,arrival=1):
    return {'request':name,'prompt':128,'generated':0,'max_tokens':1024,'arrival':arrival}
def goal(name,score,members,probability=1):
    return {'chain':name,'stage':0,'probability':probability,'remaining_work':1.,'critical_mean_seconds':1.,
            'score':score,'deadline':100.,'arrival':1.,'members':members}

class LeasePacking(unittest.TestCase):
    def test_protected_incumbent_consumes_capacity_before_new_work(self):
        gs=[goal('old',1,[member('old')]),goal('new',10,[member('new')])]
        selected,_=select(gs,12,32,protected={'old'})
        self.assertEqual(selected,{'old'})
    def test_protected_member_is_not_counted_twice_in_compound(self):
        gs=[goal('compound',1,[member('old'),member('sibling')])]
        selected,_=select(gs,24,2,protected={'old'})
        self.assertEqual(selected,{'old','sibling'})
    def test_protected_member_consumes_sequence_slot(self):
        gs=[goal('old',1,[member('old')]),goal('new',10,[member('new')])]
        self.assertEqual(select(gs,200,1,protected={'old'})[0],{'old'})
    def test_already_overcommitted_does_not_admit_more(self):
        gs=[goal('old',1,[member('old')]),goal('new',10,[member('new')])]
        self.assertEqual(select(gs,8,32,protected={'old'})[0],{'old'})
    def test_ready_work_remains_work_conserving_when_no_positive_goal(self):
        gs=[goal('old',1,[member('old')],0)]
        self.assertIsNone(select(gs,12,32,protected={'old'})[0])

class ForecastRefresh(unittest.TestCase):
    def test_late_compound_sibling_changes_neighbors_and_order_does_not(self):
        model={'neighbors':1,'prefill_tokens_per_decode_unit':256,'interstage_gap_decode_units':32,
               'forecast_guard_decode_units':50,'service_safety_factor':1.15,
               'records':[{'id':1,'stages':[{'inputs':[100,100],'outputs':[20,20]}]},
                          {'id':2,'stages':[{'inputs':[100,900],'outputs':[100,100]}]}]}
        f=StageForecaster(model)
        meta={'stages':1,'widths':[2],'observed':{},'neighbor_cache':{}}
        a={'member':0,'declared_input':100};b={'member':1,'declared_input':900}
        self.assertEqual(f.neighbors(meta,0,[a])[0]['id'],1)
        self.assertEqual(f.neighbors(meta,0,[a,b])[0]['id'],2)
        self.assertEqual(f.neighbors(meta,0,[b,a])[0]['id'],2)

class TransportIsolation(unittest.TestCase):
    def test_server_routes_only_official_local_prediction_endpoint(self):
        original=socket.socket
        with original() as listener:
            listener.bind(('127.0.0.1',0));listener.listen(1);port=listener.getsockname()[1]
            try:
                isolate_prediction_port(port,'server')
                with socket.socket() as client:
                    client.connect(('127.0.0.1',65433));client.sendall(b'unchanged-payload')
                    peer,_=listener.accept()
                    with peer:self.assertEqual(peer.recv(100),b'unchanged-payload')
                with socket.socket() as unrelated:
                    unrelated.bind(('127.0.0.1',0))
                    self.assertNotEqual(unrelated.getsockname()[1],port)
            finally:socket.socket=original
    def test_predictor_bind_uses_private_port(self):
        original=socket.socket
        with original() as temporary:
            temporary.bind(('127.0.0.1',0));port=temporary.getsockname()[1]
        try:
            isolate_prediction_port(port,'predictor')
            with socket.socket() as listener:
                listener.bind(('127.0.0.1',65433))
                self.assertEqual(listener.getsockname()[1],port)
        finally:socket.socket=original

class SequentialControllerOrder(unittest.TestCase):
    def test_gpu0_is_released_before_next_controller_starts(self):
        from unittest.mock import patch
        from run_experiments import sequential_controllers
        events=[]
        class Child:
            def __init__(self,command,**kwargs):self.command=command;events.append(('start',command[0]))
            def wait(self):events.append(('end',self.command[0]));return 0
        with patch('run_experiments.subprocess.Popen',Child):
            self.assertEqual(sequential_controllers({'A':['A'],'C':['C']},{'A':None,'C':None},'.'),{'A':0,'C':0})
        self.assertEqual(events,[('start','A'),('end','A'),('start','C'),('end','C')])

if __name__=='__main__':unittest.main()
