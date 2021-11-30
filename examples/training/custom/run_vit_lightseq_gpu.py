"""
On AWS GPU node

conda activate torch-1.10-py38
pip install lightseq fairseq sacremoses

cd /fsx/users/willfeng/repos
rm -rf ./lightseq || true
git clone https://github.com/yf225/lightseq.git -b vit_dummy_data
cd ./lightseq

# TODO: how to get DP running?
python3 run_vit_lightseq_gpu.py
"""

import torch
import time
import statistics

from transformers import BertTokenizer
from lightseq.training import LSTransformer, LSCrossEntropyLayer, LSAdam


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--micro_batch_size", type=int)
args = parser.parse_args()


class VitDummyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size, crop_size, num_classes):
        self.dataset_size = dataset_size
        self.crop_size = crop_size
        self.num_classes = num_classes

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return (torch.rand(3, self.crop_size, self.crop_size).to(torch.half), torch.randint(self.num_classes, (1,)).to(torch.long))


def create_model():
    hidden_size = 1280
    transformer_config = LSTransformer.get_config(
        model="vit-h/16",
        nhead=16,  # number of heads in attention
        hidden_size=hidden_size,  # size of transformer hidden layers
        num_encoder_layer=32,
        num_decoder_layer=0,
        intermediate_size=4*hidden_size,  # size of ffn inner size
        max_seq_len=196,
        attn_prob_dropout_ratio=0.0,  # attention score dropout ratio
        activation_dropout_ratio=0.0,  # ffn activation dropout ratio
        hidden_dropout_ratio=0.0,  # dropout ration before residual
        pre_layer_norm=True,  # pre layer norm or post
        activation_fn="gelu",  # relu or gelu
        fp16=True,
        local_rank=0,
    )
    model = LSTransformer(transformer_config)
    model.to(dtype=torch.half, device=torch.device("cuda:0"))
    return model


def create_criterion():
    ce_config = LSCrossEntropyLayer.get_config(
        epsilon=0.0,
        fp16=True,
        local_rank=0,
    )
    loss_fn = LSCrossEntropyLayer(ce_config)
    loss_fn.to(dtype=torch.half, device=torch.device("cuda:0"))
    return loss_fn


if __name__ == "__main__":
    global_batch_size = args.micro_batch_size * torch.distributed.get_world_size()
    dataset_train = VitDummyDataset(global_batch_size * 10, 224, 1000)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=global_batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    model = create_model()
    loss_fn = create_criterion()
    opt = LSAdam(model.parameters(), lr=1e-5)

    print("========================TRAIN========================")
    model.train()
    step_duration_list = []
    start_time = time.time()
    for step, batch in enumerate(dataloader_train):
        output = model(batch)
        loss, _ = loss_fn(output, target)
        loss.backward()
        opt.step()
        step_duration_list.append(time.time() - start_time)
        start_time = time.time()

    print(statistics.median(step_duration_list))
