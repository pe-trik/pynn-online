# -*- coding: utf-8 -*-
# cython: language_level=3
import numpy as np
import time
import argparse


from pynn.util import audio
from decoder import init_asr_model, decode, token2word
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from segmenter import Segmenter
from server import Server

def reset_callbacl():
    segmenter.reset()

def processing_finalize_callback():
    print("INFO in processing finalize callback")
    segmenter.reset()

def processing_error_callback():
    print("INFO In processing error callback")

def processing_break_callback():
    print("INFO in processing break callback")

def init_callback():
    segmenter.reset()
    print("INFO in processing init callback ")

def data_callback(i,data):
    try:
        s = AudioSegment(data)
    except CouldntDecodeError:
        s = AudioSegment(data, sample_width=2, frame_rate=16000, channels=1)
    sample = np.asarray(s.get_array_of_samples(), dtype=np.int16)
    segmenter.append_signal(sample.tobytes())
    return  0

parser = argparse.ArgumentParser(description='pynn')
#model argument
parser.add_argument('--model-dic', help='model dictionary', required=True)
parser.add_argument('--dict', help='dictionary file', default=None)
parser.add_argument('--device', help='device', type=str, default='cuda')
parser.add_argument('--beam-size', help='beam size', type=int, default=8)
parser.add_argument('--attn-head', help='attention head', type=int, default=0)
parser.add_argument('--attn-padding', help='attention padding', type=float, default=0.05)
parser.add_argument('--stable-time', help='stable size', type=int, default=200)
parser.add_argument('--fp16', help='float 16 bits', action='store_true')
parser.add_argument('--prune', help='pruning threshold', type=float, default=1.0)
parser.add_argument('--incl-block', help='incremental block size', type=int, default=50)
parser.add_argument('--max-len', help='max length', type=int, default=100)
parser.add_argument('--space', help='space token', type=str, default='▁')
parser.add_argument('--out-seg', help='output when audio segment is complete', action='store_true')

#worker argument
parser.add_argument('-s','--server', type=str, default="i13srv53.ira.uka.de")
parser.add_argument('-p','--port' ,type=int, default=60019)
parser.add_argument('-n','--name' ,type=str, default="asr-EN")
parser.add_argument('-fi','--fingerprint', type=str, default="en-EU")
parser.add_argument('-fo','--outfingerprint',type=str, default="en-EU")
parser.add_argument('-i','--inputType' ,type=str, default="audio")
parser.add_argument('-o','--outputType', type=str, default="unseg-text")
args = parser.parse_args()

serverHost = args.server
serverPort = args.port
worker_name = args.name
inputFingerPrint  = args.fingerprint
inputType         = args.inputType
outputFingerPrint = args.outfingerprint
outputType        = args.outputType
specifier           = ""

sample_rate = 16000
VAD_aggressive = 2
padding_duration_ms = 450
frame_duration_ms = 30
rate_begin = 0.65
rate_end = 0.55
segmenter = Segmenter(sample_rate, VAD_aggressive, padding_duration_ms, frame_duration_ms, rate_begin, rate_end)

print("#" * 40 + " >> TESTING MCLOUD WRAPPER API << " + "#" * 40)
server = Server(serverHost, serverPort, data_callback, reset_callbacl)
# mcloud_w = MCloudWrap("asr".encode("utf-8"), 1)
# mcloud_w.add_service(worker_name.encode("utf-8"), "asr".encode("utf-8"), inputFingerPrint.encode("utf-8"), inputType.encode("utf-8"),outputFingerPrint.encode("utf-8"), outputType.encode("utf-8"), specifier.encode("utf-8"))
# #set callback
# mcloud_w.set_callback("init", init_callback)
# mcloud_w.set_data_callback("worker")
# mcloud_w.set_callback("finalize", processing_finalize_callback)
# mcloud_w.set_callback("error", processing_error_callback)
# mcloud_w.set_callback("break", processing_break_callback)

#clean tempfile

fbank_mat = audio.filter_bank(sample_rate, 256, 40)
print("Initialize the model...")
model, device, dic  = init_asr_model(args)
print("Done.")

#segmentor thread
#record = threading.Thread(target=segmenter_thread, args=(mcloud_w,))
#record.start()
server.start()

try:
    while True:
        if not segmenter.ready:
            time.sleep(0.2)
            continue
        segment = segmenter.active_seg()
        if segment is None:
            time.sleep(0.1)
            continue

        ss = segment.start_time()
        ntime = 600 # 1000
        phypo, shypo = [1], [1]
        h_start, h_end = 0, 0
        while not segment.completed and not args.out_seg:
            sec = segment.size() // (16*2)
            #print('Segment size: %d miliseconds' % sec)
            if sec > ntime:
                ntime = max(ntime + 600, sec)
                adc = segment.get_all()
                print(f'decoding {segment.start_time()} {segment.size()}' )
                hypo, sp, ep, frames, attn = decode(model, device, args, adc, fbank_mat, h_start, shypo)

                print()
                print(hypo)
                print(' '.join(token2word(hypo[1:], dic, args.space)))
                print(sp)
                print(ep)
                print(frames)
                print(attn)
                print()

                if len(hypo) == 0: continue

                for j in range(len(shypo), min(len(phypo), len(hypo))):
                    if phypo[j] != hypo[j]: break
                    shypo = hypo[:j]
                if shypo[-1] == 2: shypo = shypo[:-1]
                #print(shypo)
                phypo = hypo

                end = 0 if len(hypo)<=2 else ep[len(hypo)-2]

                start = ss + h_start*10
                if h_start + end > h_end:
                    h_end = h_start + end
                    if h_end > h_start+300:
                        j = len(shypo)-1
                        while j > 2:
                            if (sp[j-1]+16) >= ep[j-2] and dic[shypo[j]-2].startswith(args.space): break
                            j -= 1
                        if j > 5:
                            h_end = h_start + sp[j-1] - 1
                            hypo = shypo[:j] + [2]
                            h_start, shypo, phypo = h_end+1, [1]+shypo[j:], [1]+phypo[j:]
                    end = ss + h_end*10
                else:
                    end = start + frames*10
                hypo = token2word(hypo[1:], dic, args.space)
                print(f'Sending {start} {end}: ' + ' '.join(hypo))
                #mcloud_w.send_packet_result_async(start, end, hypo, len(hypo))
                h = "".join([h + " " for h in hypo])
                server.send_hypothesis(f'{start} {end} {h}\n')
                # TODO: send hypo
                time.sleep(0.1)

        if segment.completed:
            adc = segment.get_all()
            print(f'decoding completed {segment.start_time()} {segment.size()}' )
            hypo, sp, ep, frames = decode(model, device, args, adc, fbank_mat, h_start, shypo)
            if len(hypo) == 0: continue
            hypo = token2word(hypo[1:], dic, args.space)
            print(f'Sending {start} {end}: ' + ' '.join(hypo))
            start, end = ss + h_start*10, ss + (h_start+frames)*10
            #mcloud_w.send_packet_result_async(start, end, hypo, len(hypo))
            h = "".join([h + " " for h in hypo])
            server.send_hypothesis(f'{start} {end} {h}\n')
            # TODO: send hypo
            
            segment.finish()
            print('Finished.')

        time.sleep(0.2)
except KeyboardInterrupt:
    print("Terminating running threads..")

#server.join()

