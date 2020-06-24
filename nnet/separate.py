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
            sp_samps = np.squeeze(sps.detach().cpu().numpy())
            return [sp_samps]


def run():
    mix_input = WaveReader("data/wsj0_2mix/tt/mix.scp", sample_rate=8000)
    aux_input = WaveReader("data/wsj0_2mix/tt/aux.scp", sample_rate=8000)
    computer = NnetComputer("exp_epoch114/conv_tasnet/conv-net", 3)
    for key, mix_samps in mix_input:
        aux_samps = aux_input[key]
        logger.info("Compute on utterance {}...".format(key))
        spks = computer.compute(mix_samps, aux_samps, len(aux_samps))
        norm = np.linalg.norm(mix_samps, np.inf)
        for idx, samps in enumerate(spks):
            samps = samps[:mix_samps.size]
            # norm
            samps = samps * norm / np.max(np.abs(samps))
            write_wav(
                os.path.join("rec/", "{}.wav".format(key)), 
                samps,
                fs=8000)
    logger.info("Compute over {:d} utterances".format(len(mix_input)))


if __name__ == "__main__":
    run()
