from models import StyleEncoder, Discriminator, MappingNetwork, Generator
import torch
import torch.nn.functional as F
from tqdm.auto import trange
import wandb
from lpips_pytorch import LPIPS
import numpy as np
from utils import save_examlpes, save_checkpoint
from torchvision import transforms

lpips = LPIPS()

def generate_latents(cfg):
    return torch.randn((cfg['batch_size'], cfg['latent_dim'])).to(cfg['device'])

def zero_all_grad(cfg):
    cfg['optimizer_style_encoder'].zero_grad()
    cfg['optimizer_discriminator'].zero_grad()
    cfg['optimizer_mapping_network'].zero_grad()
    cfg['optimizer_generator'].zero_grad()


def calculate_discriminator_loss(cfg, logs, use_latents):
    style_encoder, discriminator, mapping_network, generator = cfg['style_encoder'], cfg['discriminator'], cfg['mapping_network'], cfg['generator']
    dataloader = iter(cfg['dataloader'])
    
    with torch.no_grad():
        x_real, y_org = next(dataloader)
        x_real, y_org = x_real.to(cfg['device']), y_org.to(cfg['device'])

        if use_latents:
            y_trg = torch.randint(0, cfg['num_domains'], size=(cfg['batch_size'], ), device=cfg['device'])
            z = generate_latents(cfg)
            s_trg = mapping_network(z, y_trg)
        else:
            x_ref, y_trg = next(dataloader)
            x_ref, y_trg = x_ref.to(cfg['device']), y_trg.to(cfg['device'])
            s_trg = style_encoder(x_ref, y_trg)
        x_fake = generator(x_real, s_trg)
    x_real.requires_grad_()
    
    pred_real = discriminator(x_real, y_org)
    pred_fake = discriminator(x_fake, y_trg)
    
    # 1 - real, 0 - fake
    real_loss = F.binary_cross_entropy_with_logits(pred_real, cfg['discriminator_label_smoothing'] * torch.ones(cfg['batch_size'], device=cfg['device']))
    fake_loss = F.binary_cross_entropy_with_logits(pred_fake, (1 - cfg['discriminator_label_smoothing']) * torch.ones(cfg['batch_size'], device=cfg['device']))
    
    batch_size = x_real.shape[0]
    grad_dout = torch.autograd.grad(
        outputs=pred_real.sum(), inputs=x_real,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    grad_penalty = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    
    
    discriminator_loss = real_loss + fake_loss + grad_penalty
    logs['discriminator_real_loss'] = real_loss.item()
    logs['discriminator_fake_loss'] = fake_loss.item()
    logs['discriminator_total_loss'] = discriminator_loss.item()
    logs['grad_penalty'] = grad_penalty.item()

    return discriminator_loss


def calculate_generator_loss_latents(cfg, logs):
    style_encoder, discriminator, mapping_network, generator = cfg['style_encoder'], cfg['discriminator'], cfg['mapping_network'], cfg['generator']
    dataloader = cfg['dataloader']
    
    x_real, y_org = next(iter(dataloader))
    x_real = x_real.to(cfg['device'])
    y_org = y_org.to(cfg['device'])
    y_trg = torch.randint(0, cfg['num_domains'], size=(cfg['batch_size'], ), device=cfg['device'])
    z = generate_latents(cfg)
    z2 = generate_latents(cfg)
    s_org = style_encoder(x_real, y_org)
    s_trg = mapping_network(z, y_trg)
    s_trg_2 = mapping_network(z2, y_trg)
    x_fake = generator(x_real, s_trg)
    x_fake_2 = generator(x_real, s_trg_2)
    
    style_reconstruction_loss = torch.mean(torch.abs(style_encoder(x_fake, y_trg) - s_trg))
    preserving_loss = torch.mean(torch.abs(generator(x_fake, s_org) - x_real))
    diversity_loss = -1 * torch.mean(torch.abs(x_fake - x_fake_2))
    dsc_pred = discriminator(x_fake, y_trg) # 1 - real, 0 - fake
    generator_adv_loss = F.binary_cross_entropy_with_logits(dsc_pred, cfg['generator_label_smoothing'] * torch.ones(cfg['batch_size'], device=cfg['device']))
    
    loss = style_reconstruction_loss + generator_adv_loss + preserving_loss + diversity_loss
        
    logs['latents_style_reconstruction_loss'] = style_reconstruction_loss.item()
    logs['latents_generator_adv_loss'] = generator_adv_loss.item()
    logs['latents_preserving_loss'] = preserving_loss.item()
    logs['latents_diversity_loss'] = diversity_loss.item()
    logs['latents_total_genarator_loss'] = loss.item()
    
    if logs['iter'] % 100 == 0:
        save_examlpes(x_real, None, x_fake)
    return loss

def calculate_generator_loss_references(cfg, logs):
    style_encoder, discriminator, mapping_network, generator = cfg['style_encoder'], cfg['discriminator'], cfg['mapping_network'], cfg['generator']
    dataloader = iter(cfg['dataloader'])
    
    x_real, y_org = next(dataloader)
    x_ref, y_ref = next(dataloader)
    x_ref_2, y_ref_2 = next(dataloader)
    
    x_real, x_ref, x_ref_2 = x_real.to(cfg['device']), x_ref.to(cfg['device']), x_ref_2.to(cfg['device'])
    y_org, y_ref, y_ref_2 = y_org.to(cfg['device']), y_ref.to(cfg['device']), y_ref_2.to(cfg['device'])

    s_org = style_encoder(x_real, y_org)
    s_ref = style_encoder(x_ref, y_ref)
    s_ref_2 = style_encoder(x_ref_2, y_ref_2)
    
    x_fake = generator(x_real, s_ref)
    x_fake_2 = generator(x_real, s_ref_2)
    
    style_reconstruction_loss = torch.mean(torch.abs(style_encoder(x_fake, y_ref) - s_ref))
    preserving_loss = torch.mean(torch.abs(generator(x_fake, s_org) - x_real))
    diversity_loss = -1 * torch.mean(torch.abs(x_fake - x_fake_2))
    dsc_pred = discriminator(x_fake, y_ref) # 1 - real, 0 - fake
    generator_adv_loss = F.binary_cross_entropy_with_logits(dsc_pred, cfg['generator_label_smoothing'] * torch.ones(cfg['batch_size'], device=cfg['device']))
    
    loss = style_reconstruction_loss + generator_adv_loss + preserving_loss + diversity_loss
        
    logs['refs_style_reconstruction_loss'] = style_reconstruction_loss.item()
    logs['refs_generator_adv_loss'] = generator_adv_loss.item()
    logs['refs_preserving_loss'] = preserving_loss.item()
    logs['refs_diversity_loss'] = diversity_loss.item()
    logs['refs_total_genarator_loss'] = loss.item()
    
    if logs['iter'] % 100 == 0:
        save_examlpes(x_real, x_ref, x_fake)
    return loss
    

def train(cfg):
    for i in range(cfg['total_iters']):
        logs = {'iter': i}
        
        # train discriminator -- latents
        zero_all_grad(cfg)
        discriminator_loss = calculate_discriminator_loss(cfg, logs, use_latents=True)
        discriminator_loss.backward()
        cfg['optimizer_discriminator'].step()
        
        # train discriminator -- refs
        zero_all_grad(cfg)
        discriminator_loss = calculate_discriminator_loss(cfg, logs, use_latents=False)
        discriminator_loss.backward()
        cfg['optimizer_discriminator'].step
        
        # train generator with latents
        zero_all_grad(cfg)
        loss = calculate_generator_loss_latents(cfg, logs)
        loss.backward()
        cfg['optimizer_generator'].step()
        cfg['optimizer_mapping_network'].step()
        cfg['optimizer_style_encoder'].step()
        
        # train generator with refs
        zero_all_grad(cfg)
        loss = calculate_generator_loss_references(cfg, logs)
        loss.backward()
        cfg['optimizer_generator'].step()
        
        print(logs)
        wandb.log(logs)
        
        if logs['iter'] % 500 == 0:
            save_checkpoint(cfg, logs['iter'])
        
        if logs['iter'] % 500 == 0:
            calc_lpips(cfg)
    
def calc_lpips(cfg):
    style_encoder, discriminator, mapping_network, generator = cfg['style_encoder'], cfg['discriminator'], cfg['mapping_network'], cfg['generator']
    dataloader = iter(cfg['dataloader'])
    denorm = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
    ])
    
    test_iters = 100
    values = []
    for i in range(test_iters):
        with torch.no_grad():
            x_real, _ = next(dataloader)
            x_ref, y_ref = next(dataloader)
            x_real, x_ref, y_ref = x_real.to(cfg['device']), x_ref.to(cfg['device']), y_ref.to(cfg['device'])
             
            s = style_encoder(x_ref, y_ref)
            x_fake = generator(x_real, s)
            
            x_real = denorm(x_real.cpu()) * 2 - 1
            x_fake = denorm(x_fake.cpu()) * 2 - 1

            values.append(lpips(x_fake, x_real.cpu() * 2 - 1).squeeze().item())
    res = np.mean(values) / cfg['batch_size']
    print("lpips:", res)
    wandb.log({"lpips": res})
