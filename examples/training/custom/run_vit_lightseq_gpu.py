"""
On AWS GPU node

conda activate torch-1.10-py38
pip install lightseq fairseq sacremoses

cd /fsx/users/willfeng/repos
rm -rf ./lightseq || true
git clone https://github.com/yf225/lightseq.git -b vit_dummy_data
cd ./lightseq

# TODO: how to get DP running?
python -m torch.distributed.launch --nproc_per_node=4 \
examples/training/custom/run_vit_lightseq_gpu.py --micro_batch_size=2
"""

import torch
import time
import statistics
import os
from dataclasses import dataclass

from lightseq.training import LSCrossEntropyLayer, LSAdam
from lightseq.training.ops.pytorch.transformer import LSTransformerEncoder


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--micro_batch_size", type=int)
parser.add_argument("--local_rank", default=0, type=int)


num_attention_heads = 16
hidden_size = 1280
num_layers = 32

img_size = 224
patch_size = 16  # Size of the patches to be extract from the input images

num_classes = 1000
num_epochs = 10


class VitDummyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size, crop_size, num_classes):
        self.dataset_size = dataset_size
        self.crop_size = crop_size
        self.num_classes = num_classes

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return (torch.rand(3, self.crop_size, self.crop_size).to(torch.half), torch.randint(self.num_classes, (1,)).to(torch.long))


class PatchEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.local_rank >= 0:
            torch.cuda.set_device(config.local_rank)
        img_size = (config.img_size, config.img_size)
        patch_size = (config.patch_size, config.patch_size)
        self.patch_size = patch_size
        self.in_chans = config.in_chans
        self.flatten_dim = self.patch_size[0] * self.patch_size[1] * self.in_chans
        self.img_size = img_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection = torch.nn.Linear(
            self.flatten_dim, config.embedding_dim
        )
        self.position_embedding = torch.nn.Embedding(
            num_embeddings=self.num_patches, embedding_dim=config.embedding_dim
        )

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            embedding_dim: int
            img_size: int
            patch_size: int
            in_chans: int
            embedding_dim: int
            local_rank: int
        return Config(**kwargs)

    def forward(self, input):
        print("len(input): ", len(input))
        print("input[0].shape: ", input[0].shape)
        print("input[1].shape: ", input[1].shape)
        rearranged_input = input.view(-1, self.grid_size[0] * self.grid_size[1], self.patch_size[0] * self.patch_size[1] * self.in_chans)
        # rearranged_input = einops.rearrange(
        #     input,
        #     "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        #     p1=self.patch_size[0],
        #     p2=self.patch_size[1],
        # )
        positions = torch.arange(start=0, end=self.num_patches, step=1).to(input.device)
        ret = self.projection(rearranged_input)
        ret = ret + self.position_embedding(positions)
        return ret


class LSVisionTransformer(torch.nn.Module):
    def __init__(self, config):
        super(LSVisionTransformer, self).__init__()
        self.config = config

        print("Lightseq Transformer config is ", self.config.__dict__)

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        self.build_model(self.config)

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            max_batch_tokens: int  # max batch token numbers
            max_seq_len: int  # max sequence length
            num_encoder_layer: int  # number of encoder layer
            num_decoder_layer: int  # number of decoder layer
            hidden_size: int  # size of transformer hidden layers
            intermediate_size: int  # size of ffn inner size
            nhead: int  # number of heads in attention
            attn_prob_dropout_ratio: float  # attention score dropout ratio
            activation_dropout_ratio: float  # ffn activation dropout ratio
            hidden_dropout_ratio: float  # dropout ration before residual
            pre_layer_norm: bool  # pre layer norm or post
            activation_fn: str  # relu or gelu
            fp16: bool  # fp16 presion
            local_rank: int  # rank in local node
            img_size: int
            patch_size: int
            in_chans: int
            padding_idx: int  # value doesn't matter

        return Config(**kwargs)

    def build_model(self, config):
        encoder_embed_tokens = self.build_embedding(config)

        self.encoder = self.build_encoder(config, encoder_embed_tokens)

    def build_embedding(self, config):
        emb = PatchEncoder(
            PatchEncoder.get_config(
                img_size=config.img_size,
                patch_size=config.patch_size,
                in_chans=config.in_chans,
                embedding_dim=config.hidden_size,
                local_rank=config.local_rank,
            )
        )
        return emb

    def build_encoder(self, config, embed_tokens):
        return LSTransformerEncoder(config, embed_tokens)

    def forward(self, src_tokens):
        encoder_out, _ = self.encoder(src_tokens)
        return encoder_out


def create_model():
    hidden_size = 1280
    transformer_config = LSVisionTransformer.get_config(
        nhead=16,  # number of heads in attention
        hidden_size=hidden_size,  # size of transformer hidden layers
        num_encoder_layer=32,
        num_decoder_layer=0,
        intermediate_size=4*hidden_size,  # size of ffn inner size
        max_seq_len=(img_size // patch_size) ** 2,
        max_batch_tokens=((img_size // patch_size) ** 2) * args.micro_batch_size,
        attn_prob_dropout_ratio=0.0,  # attention score dropout ratio
        activation_dropout_ratio=0.0,  # ffn activation dropout ratio
        hidden_dropout_ratio=0.0,  # dropout ration before residual
        pre_layer_norm=True,  # pre layer norm or post
        activation_fn="gelu",  # relu or gelu
        fp16=True,
        local_rank=0,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        padding_idx=1,
    )
    model = LSVisionTransformer(transformer_config)
    model.to(dtype=torch.half, device=torch.device("cuda:0"))
    return model


def create_criterion():
    ce_config = LSCrossEntropyLayer.get_config(
        epsilon=0.0,
        fp16=True,
        local_rank=0,
        max_batch_tokens=((img_size // patch_size) ** 2) * args.micro_batch_size,
        padding_idx=0,
    )
    loss_fn = LSCrossEntropyLayer(ce_config)
    loss_fn.to(dtype=torch.half, device=torch.device("cuda:0"))
    return loss_fn


if __name__ == "__main__":
    args = parser.parse_args()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        print('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    global_batch_size = args.micro_batch_size * torch.distributed.get_world_size()
    dataset_train = VitDummyDataset(global_batch_size * 10, img_size, num_classes)
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
