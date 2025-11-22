
import os
from typing import Dict
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
# from torchvision.utils import save_signal
import dataset_diffusion
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler
from utils.extra_data_get_dataloader import get_dataloader
from torch.utils.data import ConcatDataset, DataLoader


def train(modelConfig: Dict, args: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    Dataset = getattr(dataset_diffusion, args.data_name)
    dataset = get_dataloader(args, Dataset)
    dataset = ConcatDataset(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=modelConfig["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for data in tqdmDataLoader:
                signals, labels = data
                signals = signals.view(len(signals), 1, 32, 32)
                # train
                optimizer.zero_grad()
                x_0 = signals.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisysignal = torch.randn(
            size=[1000, 1, 32, 32], device=device)
        saveNoisy = torch.clamp(noisysignal * 0.5 + 0.5, 0, 1)

        sampledSignal = sampler(noisysignal)
        sampledSignal = sampledSignal * 0.5 + 0.5  # [0 ~ 1]
        # Convert the tensor to a numpy array
        sampled_signal_np = sampledSignal.cpu().numpy()

        # Flatten the array to save it as a single column of data
        sampled_signal_flat = sampled_signal_np.flatten()

        # Define the path for the text file
        txt_file_path = os.path.join(modelConfig["sampled_dir"], "sampledSignal.txt")

        # Save the array to a text file, with each element on a new line
        np.savetxt(txt_file_path, sampled_signal_flat, fmt='%f')