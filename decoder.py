#!/usr/bin/python

import time
import subprocess
import os
import io
import re

import numpy
import torch
import torchaudio.compliance.kaldi as kaldi

from pynn.util import audio
from pynn.decoder.s2s import Beam
from pynn.util import load_object

def token2word(tokens, dic, space='', cleaning=True):
    hypo = []
    pw = ''
    for tid in tokens:
        if tid == 2:
            if pw != '': hypo.append(pw)
            break
        token = dic[tid-2]
        if space == '':
            hypo.append(token)
        else:
            if token.startswith(space):
                if pw != '': hypo.append(pw)
                pw = token.replace(space, '') if token != space else ''
            else:
                pw += token
    if cleaning:
        hypo = ['<unk>' if w.startswith('%') or w.startswith('+') or w.startswith('<') or \
                w.startswith('-') or w.endswith('-') or w.find('*')>-1 else w for w in hypo]
        words, pw = [], ''
        for w in hypo:
            if w == '<unk>' and pw == w: continue
            words.append(w)
            pw = w
        hypo = words

    return hypo

def incl_search(model, src, max_node=8, max_len=10, states=[1], len_norm=False, prune=1.0):
    enc_out = model.encode(src.unsqueeze(0), None)[0]
    enc_mask = torch.ones((1, enc_out.size(1)), dtype=torch.uint8).to(src.device)

    beam = Beam(max_node, [1], len_norm)
    if len(states) > 1:
        seq = torch.LongTensor(states).to(src.device).view(1, -1)
        logits = model.get_logit(enc_out, enc_mask, seq)
        logits = logits.squeeze(0)
        for i in range(len(states)-1):
            token = states[i+1]
            prob = logits[i][token]
            beam.advance(0, [prob], [token])
            beam.prune()

    for step in range(max_len):
        l = 1 if step == 0 else max_node
        seq = [beam.seq(k) for k in range(l)]
        seq = torch.LongTensor(seq).to(src.device)

        if l > 1:
            cache = [beam.cache[k] for k in range(l)]
            hid, cell = zip(*cache)
            hid, cell = torch.stack(hid, dim=1), torch.stack(cell, dim=1)
            hid_cell = (hid, cell)
            seq = seq[:, -1].view(-1, 1)
        else:
            hid_cell = None

        enc = enc_out.expand(seq.size(0), -1, -1)
        mask = enc_mask.expand(seq.size(0), -1)
        dec_out, hid_cell = model.decode(enc, mask, seq, hid_cell)
        
        probs, tokens = dec_out.topk(max_node, dim=1)
        probs, tokens = probs.cpu().numpy(), tokens.cpu().numpy()

        hid, cell = hid_cell
        hid = [(hid[:,k,:].clone(), cell[:,k,:].clone()) for k in range(l)]

        for k in range(l):
            prob, token, cache = probs[k], tokens[k], hid[k]
            beam.advance(k, prob, token, cache)

        beam.prune()
        if beam.done: break
    hypo, prob = beam.best_hypo()
    sth = beam.stable_hypo(prune)

    return enc_out, enc_mask, hypo, prob, sth

def init_asr_model(args):
    dic = None
    if args.dict is not None:
        dic = {}
        fin = open(args.dict, 'r')
        for line in fin:
            tokens = line.split()
            dic[int(tokens[1])] = tokens[0]

    device = torch.device(args.device)

    mdic = torch.load(args.model_dic)
    print(mdic['class'], mdic['module'], mdic['params'])
    model = load_object(mdic['class'], mdic['module'], mdic['params'])
    model = model.to(device)
    model.load_state_dict(mdic['state'])
    model.eval()
    if args.fp16: model.half()

    return model, device, dic

def decode(model, device, args, adc, fbank_mat, start=0, prefix=[1]):
    signal = numpy.frombuffer(adc[start*16*10*2:], numpy.int16)
    signal = torch.as_tensor(signal, dtype=torch.float32)
    signal = 2*(signal.min()+signal)/(signal.min()+signal.max())-1
    feats = kaldi.fbank(signal.unsqueeze(0),
                          num_mel_bins=40,
                          frame_length=25,
                          frame_shift=10,
                          subtract_mean=True,
                          sample_frequency=16000)
    feats = feats / feats.std(0, keepdims=True)
     
    frames = (feats.shape[0])
    print("Decoding for audio segment of %d frames" % frames)
    if frames < 10: return [], None, None
    
    space, beam_size, max_len = args.space, args.beam_size, args.max_len
    win, stable_time = args.incl_block, args.stable_time
    head, padding = args.attn_head, args.attn_padding

    with torch.no_grad():
        src = feats if not args.fp16 else feats.to(torch.float16)
        src = src.to(device)
        enc_out, mask, hypo, score, sth = incl_search(model, src, beam_size, max_len, prefix)

        tgt = torch.LongTensor(hypo).to(device).view(1, -1)
        attn = model.get_attn(enc_out, mask, tgt)
        attn = attn[0]
        cs = torch.cumsum(attn[head], dim=1)
        ep = cs.le(1.-padding).sum(dim=1)
        ep = ep.cpu().numpy() * 4
        sp = sp = cs.le(padding).sum(dim=1)
        sp = sp.cpu().numpy() * 4

    return hypo, sp, ep, frames, attn
