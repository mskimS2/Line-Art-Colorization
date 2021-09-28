import torch
import argparse


def get_args():
    
    p = argparse.ArgumentParser(description="reference-based sketch image colorization")
    
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--img_size", default=256, type=int)
    p.add_argument("--num_workers", default=2, type=int)
    p.add_argument("--train_mode", default="train", type=str) # train | eval
    p.add_argument("--use_g_spec", default=True, type=str) # use spectrul normalize
    p.add_argument("--use_d_spec", default=True, type=str) # use spectrul normalize
    p.add_argument("--cuda", default="cuda" if torch.cuda.is_available else "cpu")
    p.add_argument("--load_model", default=False, type=bool)
    
    # directory
    p.add_argument("--train_dir", default=r"dataset/train/images/", type=str)
    p.add_argument("--train_sket_dir", default=r"dataset/train/sketch/", type=str)
    p.add_argument("--eval_dir", default=r"dataset/valid", type=str)
    
    # save model option
    p.add_argument("--log_steps", default=100, type=int)
    p.add_argument("--sample_step", default=1, type=int)
    p.add_argument("--save_step", default=1, type=int)
    p.add_argument("--save_start", default=1, type=int)
    p.add_argument("--ckpt", default=r"save/models", type=str, help="model checkpoint")
    p.add_argument("--valid_dir", default=r"save/validation", type=str)
    
    # log
    p.add_argument("--log_step", default=1, type=int)
    
    # learning rate
    p.add_argument("--g_lr", default=1e-4, type=float)
    p.add_argument("--d_lr", default=2e-4, type=float)
    p.add_argument("--beta1", default=0.5, type=float)
    p.add_argument("--beta2", default=0.999, type=float)
    
    # noise
    p.add_argument("--noise_lower", default=-50, type=int)
    p.add_argument("--noise_upper", default=50, type=int)
    
    # loss weight
    p.add_argument("--weight_g_fake", default=1.0, type=float)
    p.add_argument("--weight_g_recon", default=30.0, type=float)
    p.add_argument("--weight_g_triplet", default=12.0, type=float)
    p.add_argument("--weight_g_perceptual", default=0.01, type=float)
    p.add_argument("--weight_g_style", default=50.0, type=float)    
    p.add_argument("--weight_d_fake", default=1.0, type=float)
    p.add_argument("--weight_d_real", default=1.0, type=float)
    
    args = p.parse_args()
    
    return args