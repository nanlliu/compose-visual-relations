import time
import random
import os

import pandas as pd
import numpy as np

import torch

import torch.multiprocessing as mp
import torch.distributed as dist

from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from torchvision.utils import make_grid

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from third_party.metrics import calculate_fid
from third_party.utils import ReplayBuffer, ReservoirBuffer, init_distributed_mode
from models import ResNetModel, ResNetCLIP
from dataset import Dataset
from inference import gen_images

from kornia import augmentation

import argparse

parser = argparse.ArgumentParser()

# Distributed training hyperparameters
parser.add_argument("--nodes", default=1, type=int, help="number of nodes for training")
parser.add_argument("--gpus", default=1, type=int, help="number of gpus per nodes")
parser.add_argument("--node_rank", default=0, type=int, help="rank of node")

# Configurations for distributed training
parser.add_argument("--master_addr", default="8.8.8.8", type=str, help="address of communicating server")
parser.add_argument("--port", default="10002", type=str, help="port of training")
parser.add_argument("--slurm", action="store_true", help="whether we are on slurm")

# Data
parser.add_argument("--numpy_data_path", type=str, help="numpy data path for training")
parser.add_argument("--clip_features_path", type=str, help="precomputed clip features path")
parser.add_argument("--dataset", choices=['clevr', 'igibson', 'blocks'])
parser.add_argument("--batch_size", default=10, type=int)
parser.add_argument("--workers", default=4, type=int)

# Model
parser.add_argument("--multiscale", action="store_true", help="whether we use a multiscale EBM")
parser.add_argument("--self_attn", action="store_true", help="whether self attention layer is used")
parser.add_argument("--buffer_size", default=10000, type=int)
parser.add_argument("--clip", action="store_true", help="whether we use CLIP to encode (only objects)")
parser.add_argument("--clip_all", action="store_true", help="whether we use CLIP to encode (the whole caption)")

# General Experiment Settings
parser.add_argument("--logdir", default="./checkpoints", help="location where log of experiments will be stored")
parser.add_argument("--exp", default="default", help="name of experiments")
parser.add_argument("--log_interval", default=10, type=int, help="log outputs every so many batches")
parser.add_argument("--save_interval", default=1000, type=int, help="save models every so many batches")
parser.add_argument("--test_interval", default=1000, type=int, help="evaluate models every so many batches")
parser.add_argument("--resume_iter", default=0, type=int, help="iteration to resume training from")

parser.add_argument("--scheduler", action="store_true")
parser.add_argument("--transform", action="store_true",
                    help="transform the image when removing from the replay/reservoir buffer")
parser.add_argument("--kl", action="store_true")
parser.add_argument("--epoch_num", default=100, type=int, help="Number of Epochs to train on")
parser.add_argument("--ensembles", default=1, type=int, help="Number of ensembles to train models with")
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--kl_coeff", default=1.0, type=int)
parser.add_argument("--cuda", action="store_true")

# Setting for MCMC sampling
parser.add_argument("--num_steps", default=60, type=int, help="Steps of gradient descent for training")
parser.add_argument("--step_lr", default=300, type=int, help="Size of steps for gradient descent")
parser.add_argument("--replay_batch", action="store_true", help="whether we use a buffer")
parser.add_argument("--reservoir", action="store_true", help="Use a reservoir of past entries")

# Architecture Settings
parser.add_argument("--filter_dim", default=128, type=int, help="number of filter for conv layers")
parser.add_argument("--im_size", default=128, type=int, help="size of training images")
parser.add_argument("--spec_norm", action="store_true", help="Whether to use spectral normalization on weights")
parser.add_argument("--norm", action="store_true", help="Use norm in models")
parser.add_argument("--alias", action="store_true", help="Use alias in models")
parser.add_argument("--square_energy", action="store_true", help="whether apply square to the energy")
parser.add_argument("--sigmoid", action="store_true", help="whether apply sigmoid to the energy")

# Conditional settings
parser.add_argument("--cond", action="store_true", help="Conditional generation with the model")
parser.add_argument('--all_step', action="store_true", help="Langevin sampling on all steps")

jitter = augmentation.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.08, hue=0.02)


def compress_x_mod(x_mod):
    x_mod = (255 * np.clip(x_mod, 0, 1)).astype(np.uint8)
    return x_mod


def decompress_x_mod(x_mod):
    x_mod = x_mod / 256 + np.random.uniform(0, 1 / 256, x_mod.shape)
    return x_mod


def sync_model(models):
    for model in models:
        for param in model.parameters():
            dist.broadcast(param.data, 0)


def ema_model(models, models_ema, mu=0.99):
    for model, model_ema in zip(models, models_ema):
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            param_ema.data = mu * param_ema.data + (1 - mu) * param.data


def average_gradients(models):
    size = float(dist.get_world_size())

    for model in models:
        for param in model.parameters():
            if param.grad is None:
                continue

            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size


def gen_image(label, FLAGS, model, im_neg, num_steps, sample=False):
    im_noise = torch.randn_like(im_neg).detach()
    im_negs_samples = []

    for i in range(num_steps):
        im_noise.normal_()

        if FLAGS.dataset in ['clevr', 'igibson', 'visual_genome', 'blocks']:
            im_neg = im_neg + 0.005 * im_noise
        else:
            raise NotImplementedError

        im_neg.requires_grad_(requires_grad=True)
        energy = model.forward(im_neg, label)

        if FLAGS.all_step:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg], create_graph=True)[0]
        else:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg])[0]

        if i == num_steps - 1:
            im_neg_orig = im_neg
            im_neg = im_neg - FLAGS.step_lr * im_grad

            im_neg_kl = im_neg_orig[:FLAGS.batch_size]

            if not sample:
                energy = model.forward(im_neg_kl, label)
                im_grad = torch.autograd.grad([energy.sum()], [im_neg_kl], create_graph=True)[0]

            im_neg_kl = im_neg_kl - FLAGS.step_lr * im_grad[:FLAGS.batch_size]
            im_neg_kl = torch.clamp(im_neg_kl, 0, 1)
        else:
            im_neg = im_neg - FLAGS.step_lr * im_grad

        im_neg = im_neg.detach()

        if sample:
            im_negs_samples.append(im_neg)

        im_neg = torch.clamp(im_neg, 0, 1)

    if sample:
        return im_neg, im_neg_kl, im_negs_samples, im_grad
    else:
        return im_neg, im_neg_kl, im_grad


def train(models, models_ema, optimizer, writer, dataloader, resume_iter, logdir, FLAGS, rank_idx, best_fid):
    if FLAGS.replay_batch:
        if FLAGS.reservoir:
            replay_buffer = ReservoirBuffer(FLAGS.buffer_size, FLAGS.transform, FLAGS.dataset, FLAGS.im_size)
        else:
            replay_buffer = ReplayBuffer(FLAGS.buffer_size, FLAGS.transform, FLAGS.dataset, FLAGS.im_size)

    curr_iterations = resume_iter
    optimizer.zero_grad()

    if FLAGS.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1)

    num_steps = FLAGS.num_steps

    device = torch.device("cuda" if FLAGS.cuda else "cpu")

    for epoch in range(FLAGS.epoch_num):
        tock = time.time()
        for data_corrupt, data, label, captions in dataloader:
            if not FLAGS.clip:
                label = label.long()

            label = label.to(device)
            data = data.permute(0, 3, 1, 2).float().contiguous()

            if curr_iterations % FLAGS.save_interval == 0:
                if FLAGS.dataset in ['clevr', 'igibson', 'visual_genome', 'blocks']:
                    data_corrupt = torch.Tensor(
                        np.random.uniform(0.0, 1.0, (FLAGS.batch_size, FLAGS.im_size, FLAGS.im_size, 3))
                    )
                    label = label[:FLAGS.batch_size]
                    data_corrupt = data_corrupt[:FLAGS.batch_size]
                else:
                    raise ValueError(f'invalid dataset: {FLAGS.dataset}!')

            data_corrupt = data_corrupt.permute(0, 3, 1, 2).float().contiguous()
            data = data.to(device)
            data_corrupt = data_corrupt.to(device)

            if FLAGS.replay_batch and len(replay_buffer) >= FLAGS.batch_size:
                replay_batch, idxs = replay_buffer.sample(data_corrupt.size(0))
                replay_batch = decompress_x_mod(replay_batch)
                replay_mask = (np.random.uniform(0, 1, data_corrupt.size(0)) > 0.001)
                data_corrupt[replay_mask] = torch.Tensor(replay_batch[replay_mask]).to(device)

            ix = random.randint(0, len(models) - 1)
            model = models[ix]

            if curr_iterations % FLAGS.save_interval == 0:
                im_neg, im_neg_kl, im_samples, x_grad = gen_image(label, FLAGS, model, data_corrupt, num_steps, sample=True)
            else:
                im_neg, im_neg_kl, x_grad = gen_image(label, FLAGS, model, data_corrupt, num_steps)

            if FLAGS.scheduler:
                scheduler.step()

            energy_pos = model.forward(data, label)
            energy_neg = model.forward(im_neg.detach(), label)

            if FLAGS.replay_batch and im_neg is not None:
                replay_buffer.add(compress_x_mod(im_neg.detach().cpu().numpy()))

            loss = energy_pos.mean() - energy_neg.mean()
            loss = loss + (torch.pow(energy_pos, 2).mean() + torch.pow(energy_neg, 2).mean())

            if FLAGS.kl:
                model.requires_grad_(False)
                loss_kl = model.forward(im_neg_kl, label)
                model.requires_grad_(True)
            else:
                loss_kl = torch.zeros(1)

            loss = loss + FLAGS.kl_coeff * loss_kl.mean()
            loss.backward()

            if FLAGS.gpus > 1:
                average_gradients(models)

            [clip_grad_norm_(model.parameters(), 0.5) for model in models]

            optimizer.step()
            optimizer.zero_grad()

            ema_model(models, models_ema, mu=0.999)

            if curr_iterations % FLAGS.log_interval == 0 and rank_idx == 0:
                tick = time.time()

                writer.add_scalar('Positive_energy_avg/train', energy_pos.mean().item(), global_step=curr_iterations)
                writer.add_scalar('Positive_energy_std/train', energy_pos.std(unbiased=False).item(), global_step=curr_iterations)
                writer.add_scalar('Negative_energy_avg/train', energy_neg.mean().item(), global_step=curr_iterations)
                writer.add_scalar('Negative_energy_std/train', energy_neg.std(unbiased=False).item(), global_step=curr_iterations)
                writer.add_scalar('Energy_diff/train', abs(energy_pos.mean().item() - energy_neg.mean().item()), global_step=curr_iterations)

                writer.add_scalar('kl_mean/train', loss_kl.mean().item(), global_step=curr_iterations)
                writer.add_scalar('x_grad/train', torch.abs(x_grad.detach().cpu()).mean().item(), global_step=curr_iterations)

                writer.add_scalar('iteration_time/train', time.time() - tock, global_step=curr_iterations)
                writer.add_scalar('replay_buffer_size/train', len(replay_buffer), global_step=curr_iterations)

                tock = tick

            if curr_iterations % FLAGS.save_interval == 0 and rank_idx == 0:
                model_path = os.path.join(logdir, "model_{}.pth".format(curr_iterations))

                ckpt = {
                    'optimizer_state_dict': optimizer.state_dict(),
                    'FLAGS': FLAGS,
                    'best_fid': best_fid,
                    'curr_iterations': curr_iterations
                }

                for i in range(FLAGS.ensembles):
                    ckpt['model_state_dict_{}'.format(i)] = models[i].state_dict()
                    ckpt['ema_model_state_dict_{}'.format(i)] = models_ema[i].state_dict()

                torch.save(ckpt, model_path)

                model.eval()
                generated_images = gen_images(
                    model, FLAGS.dataset, label, FLAGS.num_steps,
                    FLAGS.step_lr, FLAGS.im_size, FLAGS.batch_size, FLAGS.clip, FLAGS.clip_all, device
                )
                model.train()

                fid_score = calculate_fid(
                    generated_images.detach().cpu().permute((0, 2, 3, 1)).numpy(),
                    data.detach().cpu().permute((0, 2, 3, 1)).numpy(),
                    use_multiprocessing=False,
                    batch_size=FLAGS.batch_size
                )

                generated_image_grid = make_grid(generated_images.detach().cpu(), nrow=int(FLAGS.batch_size ** 0.5))
                original_image_grid = make_grid(data.detach().cpu(), nrow=int(FLAGS.batch_size ** 0.5))
                writer.add_image('Generated_images/train', generated_image_grid, global_step=curr_iterations)
                writer.add_image('Original_images/train', original_image_grid, global_step=curr_iterations)
                writer.add_text('Captions/train', pd.DataFrame(captions, columns=['Caption']).to_markdown())

                writer.add_scalar(f'FID-{data.shape[0]}/train', fid_score, global_step=curr_iterations)

                if best_fid is None or fid_score < best_fid:
                    model_path = os.path.join(logdir, "model_best.pth")
                    torch.save(ckpt, model_path)
                    best_fid = fid_score

            curr_iterations += 1

    writer.flush()


def main_single(gpu, FLAGS):
    if FLAGS.slurm:
        init_distributed_mode(FLAGS)

    os.environ['MASTER_ADDR'] = FLAGS.master_addr
    os.environ['MASTER_PORT'] = FLAGS.port

    rank_idx = FLAGS.node_rank * FLAGS.gpus + gpu
    world_size = FLAGS.nodes * FLAGS.gpus
    print("Values of args: ", FLAGS)

    if world_size > 1:
        if FLAGS.slurm:
            dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank_idx)
        else:
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=world_size,
                                    rank=rank_idx)

    if FLAGS.dataset in ['clevr', 'igibson', 'blocks']:
        if FLAGS.clip_all:
            train_dataset = Dataset(
                dataset=FLAGS.dataset, image_size=FLAGS.im_size, datasource='random',
                numpy_file_path=FLAGS.numpy_data_path, features_path=FLAGS.clip_features_path
            )
        else:
            train_dataset = Dataset(
                dataset=FLAGS.dataset, image_size=FLAGS.im_size, datasource='random',
                numpy_file_path=FLAGS.numpy_data_path, features_path=None
            )
    else:
        raise ValueError(f'dataset: {FLAGS.dataset} is invalid!')

    if FLAGS.clip:
        if FLAGS.clip_all:
            train_dataloader = DataLoader(
                train_dataset, num_workers=0, batch_size=FLAGS.batch_size,
                shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn_clip_all
            )
        else:
            train_dataloader = DataLoader(
                train_dataset, num_workers=0, batch_size=FLAGS.batch_size,
                shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn
            )
    else:
        train_dataloader = DataLoader(
            train_dataset, num_workers=8, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True
        )

    FLAGS_OLD = FLAGS

    logdir = os.path.join(FLAGS.logdir, FLAGS.exp)
    best_fid = None

    if FLAGS.resume_iter != 0:
        model_path = os.path.join(logdir, "model_{}.pth".format(FLAGS.resume_iter))
        checkpoint = torch.load(model_path)
        best_fid = checkpoint['best_fid']
        FLAGS = checkpoint['FLAGS']

        FLAGS.resume_iter = FLAGS_OLD.resume_iter
        FLAGS.nodes = FLAGS_OLD.nodes
        FLAGS.gpus = FLAGS_OLD.gpus
        FLAGS.node_rank = FLAGS_OLD.node_rank
        FLAGS.master_addr = FLAGS_OLD.master_addr
        FLAGS.train = FLAGS_OLD.train
        FLAGS.num_steps = FLAGS_OLD.num_steps
        FLAGS.step_lr = FLAGS_OLD.step_lr
        FLAGS.batch_size = FLAGS_OLD.batch_size
        FLAGS.ensembles = FLAGS_OLD.ensembles
        FLAGS.kl_coeff = FLAGS_OLD.kl_coeff
        FLAGS.save_interval = FLAGS_OLD.save_interval

        for key in dir(FLAGS):
            if "__" not in key:
                FLAGS_OLD[key] = getattr(FLAGS, key)

        FLAGS = FLAGS_OLD

    if FLAGS.clip:
        model_fn = ResNetCLIP
    else:
        model_fn = ResNetModel

    models = [model_fn(FLAGS).train() for _ in range(FLAGS.ensembles)]
    models_ema = [model_fn(FLAGS).train() for _ in range(FLAGS.ensembles)]

    if FLAGS.cuda:
        torch.cuda.set_device(gpu)
        models = [model.cuda(gpu) for model in models]
        models_ema = [model_ema.cuda(gpu) for model_ema in models_ema]

    parameters = []
    for model in models:
        parameters.extend(list(model.parameters()))

    optimizer = Adam(parameters, lr=FLAGS.lr, betas=(0.0, 0.9), eps=1e-8)

    if FLAGS.gpus > 1:
        sync_model(models)

    ema_model(models, models_ema, mu=0.0)

    writer = SummaryWriter(f"runs/Our_LR_{FLAGS.lr}_BATCH_{FLAGS.batch_size}_STEP_SIZE_{FLAGS.step_lr}"
                           f"_DATA_{FLAGS.dataset}_CLIP_{FLAGS.clip}_CLIPALL_{FLAGS.clip_all}_{FLAGS.im_size}")

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.resume_iter != 0:
        model_path = os.path.join(logdir, "model_{}.pth".format(FLAGS.resume_iter))
        print('loading', model_path)
        checkpoint = torch.load(model_path, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for i, (model, model_ema) in enumerate(zip(models, models_ema)):
            model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)])
            model_ema.load_state_dict(checkpoint['ema_model_state_dict_{}'.format(i)])

    pytorch_total_params = sum([p.numel() for model in models for p in model.parameters() if p.requires_grad])
    print("Number of parameters for models", pytorch_total_params)

    train(models, models_ema, optimizer, writer, train_dataloader, FLAGS.resume_iter, logdir, FLAGS, rank_idx, best_fid)


def main():
    FLAGS = parser.parse_args()
    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(0, FLAGS)


if __name__ == "__main__":
    main()
