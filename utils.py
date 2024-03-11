import torch 
from torchvision import transforms
import wandb
import numpy as np
from datetime import datetime

def collate_fn(data):
    images = []
    labels = []
    for el in data:
        images.append(el[0])
        labels.append(el[1][20])
    return torch.stack(images), torch.stack(labels)


def save_examlpes(real, ref, fake):
    denorm = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
    ])
    real = denorm(real).detach().cpu().numpy() 
    fake = denorm(fake).detach().cpu().numpy() 
    
    if ref is not None:
        ref = denorm(ref).detach().cpu().numpy() 
        images = np.concatenate((real, ref, fake), axis=3) # B, C, H, 3 * W
    else:
        images = np.concatenate((real, fake), axis=3) # B, C, H, 2 * W
    
    images = [images[i, :, :, :].transpose(1,2,0) for i in range(images.shape[0])]
    if ref is None:
        wandb.log({"examples latent": [wandb.Image(img) for img in images]})
    else:
        wandb.log({"examples": [wandb.Image(img) for img in images]})
    
    
def save_checkpoint(cfg, iters):
    path = f'/home/ubuntu/DeepGenerativeModels/homeworks/second/saved/{datetime.now().strftime("%H_%M_%S")}__iter_{iters}.pt'
    
    torch.save({
            'generator_state_dict': cfg['generator'].state_dict(),
            'style_encoder_state_dict': cfg['style_encoder'].state_dict(),
            'discriminator_state_dict': cfg['discriminator'].state_dict(),
            'mapping_network_state_dict': cfg['mapping_network'].state_dict(),
            }, path)