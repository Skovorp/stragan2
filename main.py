import sys; sys.path.append('../../utils/')
# from datasets.celeba import CelebADataset
from models import StyleEncoder, Discriminator, MappingNetwork, Generator
from torchvision.datasets import CelebA
import torch
import torch.nn.functional as F
# from lpips_pytorch import LPIPS
from torchvision import transforms
from munch import Munch
import gc
from tqdm.auto import trange
from trainer import train
import yaml
from dotenv import load_dotenv
import wandb
from torch.optim import Adam
from utils import collate_fn
load_dotenv()


with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
run = wandb.init(project="StarGAN2", config=cfg, 
                #  mode="disabled"
                 )
cfg['device'] = torch.device('cuda')

# Transformations to be applied to each individual image sample
transform=transforms.Compose([
    transforms.Resize(cfg['img_size']),
    transforms.CenterCrop(cfg['img_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

dataset = CelebA(root='.', download=True, transform=transform)
cfg['dataloader'] = torch.utils.data.DataLoader(dataset,
                                        batch_size=cfg['batch_size'],
                                        num_workers=2,
                                        pin_memory=True,
                                        shuffle=True,
                                        collate_fn=collate_fn)

cfg['style_encoder'] = StyleEncoder(**cfg).to(cfg['device'])
cfg['discriminator'] = Discriminator(**cfg).to(cfg['device'])
cfg['mapping_network'] = MappingNetwork(**cfg).to(cfg['device'])
cfg['generator'] = Generator(**cfg).to(cfg['device'])

cfg['optimizer_style_encoder'] = Adam(cfg['style_encoder'].parameters(), **cfg['optimizer'])
cfg['optimizer_discriminator'] = Adam(cfg['discriminator'].parameters(), **cfg['optimizer'])
cfg['optimizer_mapping_network'] = Adam(cfg['mapping_network'].parameters(), lr=cfg['lr_mapping'], betas=cfg['optimizer']['betas'], weight_decay=cfg['optimizer']['weight_decay'])
cfg['optimizer_generator'] = Adam(cfg['generator'].parameters(), **cfg['optimizer'])

train(cfg)
