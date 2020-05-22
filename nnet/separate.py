#!/usr/bin/env python

# wujian@2018

import os
import argparse

import torch as th
import numpy as np

from conv_tas_net_decode import ConvTasNet

from libs.utils import load_json, get_logger
from libs.audio import WaveReader, write_wav
from libs.kaldi_io import read_vec_flt, read_mat
logger = get_logger(__name__)

#from pudb import set_trace
#set_trace()

class NnetComputer(object):
    def __init__(self, cpt_dir, gpuid):
        self.device = th.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else th.device("cpu")
        nnet = self._load_nnet(cpt_dir)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()

    def _load_nnet(self, cpt_dir):
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = ConvTasNet(**nnet_conf)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        return nnet

    def compute(self, samps, aux_samps, aux_samps_len):
        with th.no_grad():
            raw = th.tensor(samps, dtype=th.float32, device=self.device)
            aux = th.tensor(aux_samps, dtype=th.float32, device=self.device)
            aux_len = th.tensor(aux_samps_len, dtype=th.float32, device=self.device)
            aux = aux.unsqueeze(0)
            sps,sps2,sps3,spk_pred = self.nnet(raw, aux, aux_len)
            #sp_samps = [np.squeeze(s.detach().cpu().numpy()) for s in sps]
            sp_samps = np.squeeze(sps.detach().cpu().numpy())
            return [sp_samps]


def run():
    #mix_input = WaveReader(args.input, sample_rate=args.fs)
    mix_input = WaveReader("/export/home/clx214/gm/ntu_project/SpEx_SincNetAuxCNNEncoder_MultiOriEncoder_share_min_2spk/data/wsj0_2mix/tt/mix.scp", sample_rate=8000)
    aux_input = WaveReader("/export/home/clx214/gm/ntu_project/SpEx_SincNetAuxCNNEncoder_MultiOriEncoder_share_min_2spk/data/wsj0_2mix/tt/aux.scp", sample_rate=8000)
    computer = NnetComputer("/export/home/clx214/gm/ntu_project/SpEx_SincNetAuxCNNEncoder_MultiOriEncoder_share_min_2spk/exp_epoch114/conv_tasnet/conv-net", 0)
    #cmvn = np.load("/export/home/clx214/gm/ntu_project/SpEx2/data/tr_cmvn.npz")
    #mean_val = cmvn['mean_inputs']
    #std_val = cmvn['stddev_inputs']
    for key, mix_samps in mix_input:
        #print(key)
        #print(mix_samps)
        #spk_key = "spk_" + key.split('_')[-1][0:3]
        #aux_mfcc = read_mat(aux_input.index_dict[key.split('_')[-1]])
        #aux_mfcc = (aux_mfcc - mean_val) / (std_val + 1e-8)
        #aux_samps = read_vec_flt(aux_input.index_dict[spk_key])
        aux_samps = aux_input[key]
        logger.info("Compute on utterance {}...".format(key))
        spks = computer.compute(mix_samps, aux_samps, len(aux_samps))
        norm = np.linalg.norm(mix_samps, np.inf)
        for idx, samps in enumerate(spks):
            samps = samps[:mix_samps.size]
            # norm
            samps = samps * norm / np.max(np.abs(samps))
            write_wav(
                os.path.join("/export/home/clx214/gm/ntu_project/SpEx_SincNetAuxCNNEncoder_MultiOriEncoder_share_min_2spk/rec/", "spk{}/{}.wav".format(
                    idx + 1, key)),
                samps,
                fs=8000)
    logger.info("Compute over {:d} utterances".format(len(mix_input)))


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(
    #    description=
    #    "Command to do speech separation in time domain using ConvTasNet",
    #    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument("checkpoint", type=str, help="Directory of checkpoint")
    #parser.add_argument(
    #    "--input", type=str, required=True, help="Script for input waveform")
    #parser.add_argument(
    #    "--gpu",
    #    type=int,
    #    default=-1,
    #    help="GPU device to offload model to, -1 means running on CPU")
    #parser.add_argument(
    #    "--fs", type=int, default=8000, help="Sample rate for mixture input")
    #parser.add_argument(
    #    "--dump-dir",
    #    type=str,
    #    default="sps_tas",
    #    help="Directory to dump separated results out")
    #args = parser.parse_args()
    #run(args)
    run()
