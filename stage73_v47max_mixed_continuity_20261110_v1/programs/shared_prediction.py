"""Opt-in shared comparison runtime, installed identically for A/Astar/v48.

No predictor weights, deadline/SLO or native reward formula are changed. The
allocation path refreshes its old snapshot; a new prediction invalidates the
whole collection's cached priority. Foreground remains asynchronous.
"""
from collections import Counter
import atexit
import functools
import json
from pathlib import Path
import socket
import threading
import time
import uuid
from prediction_transport import PROTOCOL, send_frame, recv_frame

VERSION = 'shared_framed_dynamic_prediction_v1'


def install(server, policy_class, output=None, *, method, port,
            predictor_protocol, timeout_seconds=10.):
    if method not in ['A','Astar','v48','v49','v49pro','v50','v50pro','v47pro'] or predictor_protocol != PROTOCOL:
        raise ValueError('Explicit method and matching predictor protocol required')
    if not isinstance(port,int) or not 1024 <= port <= 65535 or timeout_seconds <= 0:
        raise ValueError('Invalid local prediction endpoint/timeout')
    if getattr(policy_class,'_shared_v48_prediction',False):
        raise RuntimeError('Shared prediction runtime already installed in this process')
    counts = Counter()
    lock = threading.Lock()
    closed = False
    output = Path(output) if output is not None else None
    stream = (output/'SHARED_PREDICTION_EVENTS.jsonl').open('x') if output else None
    def log(kind, **fields):
        with lock:
            counts[kind] += 1
            if stream and not closed:
                stream.write(json.dumps({'kind':kind,'monotonic':time.perf_counter(),**fields})+'\n')
                stream.flush()
    def update(info, prompt):
        identity = uuid.uuid4().hex
        fields = {'id':identity,'collection_id':info.collection_id,
                  'stage_id':info.stage_id,'request_id':info.request_id}
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=timeout_seconds) as conn:
                send_frame(conn,{'protocol':PROTOCOL,'id':identity,'prompt':[prompt]})
                result=recv_frame(conn)
            value=result['output_len']
            if result.get('protocol')!=PROTOCOL or result.get('id')!=identity or len(value)!=1 or not isinstance(value[0],int) or value[0]<1:
                raise ValueError('Mismatched or invalid predictor response')
            info.output_len=value[0]
            info.prediction_status='ready'
            info.prediction_revision=getattr(info,'prediction_revision',0)+1
            log('prediction_success',prediction=value[0],**fields)
        except Exception as error:
            # Original fallback, now explicit and never labelled a valid QRF prediction.
            info.output_len=1024; info.prediction_status='fallback_error'
            info.prediction_revision=getattr(info,'prediction_revision',0)+1
            log('prediction_fallback',error_type=type(error).__name__,error=str(error),**fields)
    server.send_and_update_request_info=update
    old_allocation=policy_class.allocation_control
    @functools.wraps(old_allocation)
    def allocation(self, seq_group):
        group=seq_group
        for peer in self.seq_group_dict[group.collection_id]:
            peer.predict_output_length=peer.request_info.output_len
        return old_allocation(self,group)
    old_priority=policy_class.get_priority
    @functools.wraps(old_priority)
    def priority(self,seq_group):
        group=seq_group
        peers=self.seq_group_dict[group.collection_id]
        signature=tuple(sorted((p.request_id,getattr(p.request_info,'prediction_revision',0))for p in peers))
        revisions=getattr(self,'_prediction_revisions',None)
        if revisions is None:self._prediction_revisions=revisions={}
        if revisions.get(group.collection_id)!=signature:
            for peer in peers:
                self.seq_group_slo_dict.pop(peer.request_id,None)
            revisions[group.collection_id]=signature
        return old_priority(self,group)
    old_delete=policy_class.delete_seq_group
    @functools.wraps(old_delete)
    def delete(self,seq_group):
        group=seq_group
        result=old_delete(self,group)
        if not self.seq_group_dict.get(group.collection_id):
            getattr(self,'_prediction_revisions',{}).pop(group.collection_id,None)
        return result
    policy_class.allocation_control=allocation
    policy_class.get_priority=priority
    policy_class.delete_seq_group=delete
    policy_class._shared_v48_prediction=True
    receipt={'version':VERSION,'method':method,'protocol':PROTOCOL,'port':port,
             'timeout_seconds':timeout_seconds,'fallback':1024,'QRF_weights_changed':False,
             'native_reward_formula_changed':False,'asynchronous':True,
             'comparison_requirement':'same module, protocol, weights and configuration for A/Astar/v48; method label alone may differ'}
    def finish():
        nonlocal closed
        with lock:
            if closed:return
            closed=True
            if stream:stream.close()
            if output and output.exists():(output/'SHARED_PREDICTION_RUNTIME.json').write_text(json.dumps({**receipt,'counts':dict(counts)},indent=2)+'\n')
    if output:
        (output/'SHARED_PREDICTION_RUNTIME.json').write_text(json.dumps(receipt,indent=2)+'\n')
    atexit.register(finish)
    return receipt
