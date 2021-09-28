import os
import torch
import torch.nn as nn

import loss_manager
from args import get_args
from trainer import Trainer
from data_loader import get_loader
from model import Generator, Discriminator


def make_train_directory(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, args.log_dir)):
        os.makedirs(os.path.join(args.save_dir, args.log_dir))
    if not os.path.exists(os.path.join(args.save_dir, args.sample_dir)):
        os.makedirs(os.path.join(args.save_dir, args.sample_dir))
    if not os.path.exists(os.path.join(args.save_dir, args.result_dir)):
        os.makedirs(os.path.join(args.save_dir, args.result_dir))
    if not os.path.exists(os.path.join(args.save_dir, args.model_dir)):
        os.makedirs(os.path.join(args.save_dir, args.model_dir))

def main(args):
    
    args = get_args()
    G = Generator(spec=args.use_g_spec)
    D = Discriminator(spec=args.use_d_spec)
    g_optimizer = torch.optim.Adam(G.parameters(), args.g_lr, (args.beta1, args.beta2))
    d_optimizer = torch.optim.Adam(D.parameters(), args.d_lr, (args.beta1, args.beta2))
    model = {'G':G, 'D':D}
    optim = {'G_optim':g_optimizer, 'D_optim':d_optimizer}
    
    SCTFLoss = loss_manager.SCTFLoss(args=args, 
                                     adversarial_loss=nn.MSELoss()) # lsgan loss
    trainer = Trainer(model=model, 
                      args=args, 
                      optim=optim,
                      data_loader=get_loader(args),
                      SCTFLoss=SCTFLoss)
    
    trainer.train()


if __name__ == '__main__':
    args = get_args()
    main(args)