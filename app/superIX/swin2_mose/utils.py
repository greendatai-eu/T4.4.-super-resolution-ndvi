import torch
import yaml

from superIX.swin2_mose.model import Swin2MoSE


def to_shape(t1, t2):
    t1 = t1[None].repeat(t2.shape[0], 1)
    t1 = t1.view((t2.shape[:2] + (1, 1)))
    return t1


def norm(tensor, mean, std):
    # get stats
    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)
    # denorm
    return (tensor - to_shape(mean, tensor)) / to_shape(std, tensor)


def denorm(tensor, mean, std):
    # get stats
    mean = torch.tensor(mean).to(tensor.device)
    std = torch.tensor(std).to(tensor.device)
    # denorm
    return (tensor * to_shape(std, tensor)) + to_shape(mean, tensor)


def load_config(path):
    # load config
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_swin2_mose(model_weights, cfg):
    # load checkpoint
    checkpoint = torch.load(model_weights, map_location='cpu')

    # build model
    sr_model = Swin2MoSE(**cfg['super_res']['model'])
    sr_model.load_state_dict(
        checkpoint['model_state_dict'])

    sr_model.cfg = cfg

    return sr_model


def run_swin2_mose(model, lr, hr, device='cuda'):

    cfg = model.cfg

    # norm fun
    hr_stats = cfg['dataset']['stats']['tensor_05m_b2b3b4b8']
    lr_stats = cfg['dataset']['stats']['tensor_10m_b2b3b4b8']

    # select 10m lr bands: B02, B03, B04, B08 and hr bands
    lr_orig = torch.from_numpy(lr)[None].float()[:, [0, 1, 2, 3]].to(device)
    hr_orig = torch.from_numpy(hr)[None].float().to(device)

    # normalize data
    lr = norm(lr_orig, mean=lr_stats['mean'], std=lr_stats['std'])
    hr = norm(hr_orig, mean=hr_stats['mean'], std=hr_stats['std'])

    # predict a image
    with torch.no_grad():
        sr = model(lr)
        if not torch.is_tensor(sr):
            sr, _ = sr

    # denorm sr
    sr = denorm(sr, mean=hr_stats['mean'], std=hr_stats['std'])    

    # Prepare output
    sr = sr.round().cpu().numpy().astype('uint16').squeeze()
    hr = hr_orig[0].cpu().numpy().astype('uint16').squeeze()

    # Use nn interpolation to go back to x2 without distortion
    # during metrics calculation
    if sr.shape[1] != hr.shape[1]:
        sr = torch.nn.functional.interpolate(
            torch.from_numpy(sr)[None].float(),
            size=hr.shape[1:],
            mode='nearest'
        ).squeeze().numpy().astype('uint16')

    
    return {
        'sr': sr,
    }
