import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import logging
import csv
import random


from torch.nn import functional as F

from pytorch_generative.models import base
from pytorch_generative.models.vae import vaes



logging.basicConfig(level=logging.INFO)
import argparse

from linear_models import (
    VectorQuantizer,
    Encoder,
    Decoder,
    VQVAEModel,
    VQVAE2Model
)

from utils import (
    load_dataset,
    set_seed_everywhere,
)

gfile = tf.io.gfile


def train(args):
    device = torch.device("cuda")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed_everywhere(args.seed)

    ## fixed dataset
    observations, actions, data_variance = load_dataset(
        args.env,
        1,
        args.datapath,
        args.normal,
        args.num_data,
        args.stack,
        args.num_episodes,
    )

    ## Stage 1
    logging.info("Building models..")
    logging.info("Start stage 1...")


    vqvae2 = VQVAE2Model( in_channels= args.stack,
                        bottom_dim = 64, 
                        top_dim = 64,
                        out_channels=args.stack,
                        hidden_channels=args.num_hiddens,
                        n_residual_blocks=args.num_residual_layers,
                        residual_channels=args.num_residual_hiddens,
                        num_embeddings=args.num_embeddings,
                        embedding_dim=args.embedding_dim,
                        commitment_cost = args.commitment_cost ).to(device)


    n_batch = len(observations) // args.batch_size + 1
    total_idxs = list(range(len(observations)))

    logging.info("Training starts..")

    save_dir = "models_vqvae2"
    if args.num_episodes is None:
        save_tag = "{}_s{}_data{}k_con{}_seed{}_ne{}_c{}".format(
            args.env,
            args.stack,
            int(args.num_data / 1000),
            1 - int(args.normal),
            args.seed,
            args.num_embeddings,
            args.commitment_cost,
        )
    else:
        save_tag = "{}_s{}_epi{}_con{}_seed{}_ne{}_c{}".format(
            args.env,
            args.stack,
            int(args.num_episodes),
            1 - int(args.normal),
            args.seed,
            args.num_embeddings,
            args.commitment_cost,
        )

    if args.add_path is not None:
        save_dir = save_dir + "_" + args.add_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ## Multi-GPU
    if torch.cuda.device_count() > 1:
        vqvae2 = nn.DataParallel(vqvae2)
    vqvae2_optimizer = torch.optim.Adam(vqvae2.parameters(), lr=args.lr)

    f = open(os.path.join(save_dir, save_tag + "_vqvae2_train.csv"), "w")
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Recon Error", "VQ Loss"])
    for epoch in tqdm(range(args.n_epochs)):
        random.shuffle(total_idxs)
        recon_errors = []
        vq_losses = []
        vqvae2.train()
        for j in range(n_batch):
            batch_idxs = total_idxs[j * args.batch_size : (j + 1) * args.batch_size]
            xx = torch.as_tensor(
                observations[batch_idxs], device=device, dtype=torch.float32
            )
            xx = xx / 255.0

            vqvae2_optimizer.zero_grad()

            #z, x_recon, vq_loss, quantized, _ = vqvae2(xx)
            encoded_b, encoded_t, x_recon, vq_loss, quantized_b, quantized_t, _, _ = vqvae2(xx)
            
            
            vq_loss = vq_loss.mean()
            recon_error = torch.mean((x_recon - xx) ** 2) / data_variance
            loss = recon_error +  vq_loss
            loss.backward()

            vqvae2_optimizer.step()

            recon_errors.append(recon_error.mean().detach().cpu().item())
            vq_losses.append(vq_loss.mean().detach().cpu().item())
        logging.info(
            "(Train) Epoch {} | Recon Error: {:.4f} | VQ Loss: {:.4f}".format(
                epoch + 1, np.mean(recon_errors), np.mean(vq_losses)
            )
        )
        writer.writerow([epoch + 1, np.mean(recon_errors), np.mean(vq_losses)])

        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                vqvae2.module.state_dict()
                if (torch.cuda.device_count() > 1)
                else vqvae2.state_dict(),
                os.path.join(save_dir, save_tag + "_ep{}_vqvae2.pth".format(epoch + 1)),
            )
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Seed & Env
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--env", default="Pong", type=str)
    parser.add_argument("--datapath", default="/data", type=str)
    parser.add_argument("--num_data", default=50000, type=int)
    parser.add_argument("--stack", default=1, type=int)
    parser.add_argument("--normal", action="store_true", default=False)

    # Save & Evaluation
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--num_episodes", default=None, type=int)
    parser.add_argument("--n_epochs", default=1000, type=int)
    parser.add_argument("--add_path", default=None, type=str)

    # VQVAE2 & Hyperparams
    parser.add_argument("--embedding_dim", default=64, type=int)
    parser.add_argument("--num_embeddings", default=512, type=int)
    parser.add_argument("--num_hiddens", default=128, type=int)
    parser.add_argument("--num_residual_layers", default=2, type=int)
    parser.add_argument("--num_residual_hiddens", default=32, type=int)
    parser.add_argument("--commitment_cost", default=0.25, type=float)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--bottom_dim", default=64, type=int)
    parser.add_argument("--top_dim", default=64, type=int)
    

    args = parser.parse_args()

    train(args)


