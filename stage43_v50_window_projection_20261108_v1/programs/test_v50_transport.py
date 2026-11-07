import io,pickle,socket,struct,threading,unittest
from types import SimpleNamespace as NS
from prediction_transport import PROTOCOL,send_frame,recv_frame,handle
from shared_prediction import install


class Fragmented:
    def __init__(self,data):self.data=io.BytesIO(data)
    def recv(self,n):return self.data.read(min(n,3))


class Wire(unittest.TestCase):
    def test_large_prompt_and_partial_header_body(self):
        value={'prompt':['x'*20000]};body=pickle.dumps(value)
        self.assertEqual(recv_frame(Fragmented(struct.pack('!I',len(body))+body)),value)

    def test_incomplete_and_oversized_frames_rejected(self):
        with self.assertRaises(EOFError):recv_frame(Fragmented(b'\x00\x00'))
        with self.assertRaises(EOFError):recv_frame(Fragmented(struct.pack('!I',10)+b'x'))
        with self.assertRaises(ValueError):recv_frame(Fragmented(struct.pack('!I',100)),maximum=10)

    def test_native_predict_function_receives_unchanged_large_prompt(self):
        a,b=socket.socketpair();self.addCleanup(a.close);self.addCleanup(b.close)
        prompt='large input '*3000;seen=[]
        def worker():
            handle(b,lambda prompts:(seen.extend(prompts)or[123]))
        t=threading.Thread(target=worker);t.start()
        send_frame(a,{'protocol':PROTOCOL,'id':'same','prompt':[prompt]});response=recv_frame(a);t.join(2)
        self.assertFalse(t.is_alive());self.assertEqual(seen,[prompt]);self.assertEqual(response['id'],'same');self.assertEqual(response['output_len'],[123])

    def test_shared_client_success_and_failure_status_are_distinct(self):
        class Policy:
            def allocation_control(self,g):return sum(p.predict_output_length for p in self.seq_group_dict[g.collection_id])
            def get_priority(self,g):return self.seq_group_slo_dict.setdefault(g.request_id,(-g.request_info.output_len,0))
            def delete_seq_group(self,g):self.seq_group_dict.pop(g.collection_id,None)
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);self.addCleanup(listener.close)
        server=NS();receipt=install(server,Policy,method='v50',port=listener.getsockname()[1],predictor_protocol=PROTOCOL,timeout_seconds=.1)
        def worker():
            conn,_=listener.accept()
            with conn:handle(conn,lambda prompts:[17])
        thread=threading.Thread(target=worker);thread.start()
        info=NS(collection_id=1,stage_id=0,request_id=0,output_len=1024)
        server.send_and_update_request_info(info,'hello'*3000);thread.join(2)
        self.assertEqual(info.output_len,17);self.assertEqual(info.prediction_status,'ready')
        p=Policy();group=NS(collection_id=1,request_id='one',request_info=info,predict_output_length=1024)
        p.seq_group_dict={1:[group]};p.seq_group_slo_dict={'one':(-1024,0)}
        self.assertEqual(p.allocation_control(group),17)
        self.assertEqual(p.get_priority(group),(-17,0))
        info.output_len=9;info.prediction_revision+=1
        self.assertEqual(p.get_priority(group),(-9,0));self.assertEqual(p.allocation_control(group),9)
        # Wrong protocol is rejected before installing a mixed client/server pair.
        with self.assertRaises(ValueError):install(NS(),type('P',(Policy,),{}),method='A',port=12345,predictor_protocol='legacy')
        listener.close()
        server.send_and_update_request_info(info,'now-unavailable')
        self.assertEqual(info.output_len,1024);self.assertEqual(info.prediction_status,'fallback_error')


if __name__=='__main__':unittest.main(verbosity=2)
