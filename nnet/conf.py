fs = 8000
chunk_len = 4  # (s)
chunk_size = chunk_len * fs
num_spks = 1

# network configure
nnet_conf = {
    "L": 20,
    "N": 256,
    "X": 8,
    "R": 4,
    "B": 256,
    "H": 512,
    "P": 3,
    "norm": "gLN",
    "num_spks": num_spks,
    "non_linear": "relu"
}

# data configure:
train_dir = "data/wsj0_2mix/tr/"
dev_dir = "data/wsj0_2mix/cv/"
spk_list = "data/wsj0_2mix_extr_tr.spk"

train_data = {
    "mix_scp":
    train_dir + "mix.scp",
    "ref_scp":
    train_dir + "ref.scp",
    "aux_scp":
    train_dir + "aux.scp",
    "spk_list": spk_list,
    "sample_rate":
    fs,
}

dev_data = {
    "mix_scp": 
    dev_dir + "mix.scp",
    "ref_scp":
    dev_dir + "ref.scp",
    "aux_scp":
    dev_dir + "aux.scp",
    "spk_list": spk_list,
    "sample_rate": fs,
}

# trainer config
adam_kwargs = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "patience": 2,
    "factor": 0.5,
    "logging_period": 200  # batch number
}
