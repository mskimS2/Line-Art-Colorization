import os
import time
import glob
import torch
import datetime
from tqdm import tqdm


class Trainer(object):

    def __init__(self, model, optim, args, data_loader, SCTFLoss):
        self.data_loader = data_loader
        self.args = args
        self.SCTFLoss = SCTFLoss
        self.G = model['G']
        self.D = model['D']
        self.g_optimizer = optim['G_optim']
        self.d_optimizer = optim['D_optim']
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

    def linear_decay_lr(self, epoch):
        decay_g_lr = self.args.g_lr - (self.args.g_lr / (self.args.epochs - 100))*(epoch-100)
        self.g_optimizer.param_groups[0]['lr'] = decay_g_lr
        decay_d_lr = self.args.d_lr - (self.args.d_lr / (self.args.epochs - 100))*(epoch-100)
        self.d_optimizer.param_groups[0]['lr'] = decay_d_lr

    def load_model(self):
        ckpt_list = glob.glob(os.path.join(self.args.ckpt, '*-gen.ckpt'))
        if len(ckpt_list) == 0:
            return 0

        ckpt_list = [int(x.split('\\')[-1].split('-')[0]) for x in ckpt_list]
        ckpt_list.sort()
        epoch = ckpt_list[-1]
        G_path = os.path.join(self.args.ckpt, f'{epoch}-gen.ckpt')
        D_path = os.path.join(self.args.ckpt, f'{epoch}-disc.ckpt')
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        
        return epoch

    def save_model(self, epoch):
        G_path = os.path.join(self.args.ckpt, f'{epoch+1}-gen.ckpt')
        D_path = os.path.join(self.args.ckpt, f'{epoch+1}-disc.ckpt')
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print(f'epoch={epoch} | Saved model')

    def train_one_epoch(self, epoch):
        start_time = time.time()
        
        data_loader = tqdm(self.data_loader, leave=True, ncols=100)
        iterations = len(self.data_loader)
        data_iter = iter(data_loader)
        print(f'iterations: {iterations} start training...')
        for i in range(iterations):
            Ir, Igt, Is = next(data_iter)
            Ir = Ir.to(self.args.cuda)
            Igt = Igt.to(self.args.cuda)
            Is = Is.to(self.args.cuda)
            
            # --------------------------------------------
            #             1. Train Discriminator
            # --------------------------------------------
            fake_images, _ = self.G(Ir, Is)
            real_score = self.D(torch.cat([Igt, Is], dim=1))
            fake_score = self.D(torch.cat([fake_images.detach(), Is], dim=1))
            
            d_loss_real, d_loss_fake = self.SCTFLoss.disc_adversarial_loss(fake_score, real_score)
            d_loss = (
                d_loss_real * self.args.weight_d_real
                + d_loss_fake * self.args.weight_d_fake
            )
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # --------------------------------------------
            #             2. Train Generator
            # --------------------------------------------
            fake_images, qkv_list = self.G(Ir, Is)
            fake_score = self.D(torch.cat([fake_images, Is], dim=1))
            
            g_loss_fake = self.SCTFLoss.gen_adversarial_loss(fake_score)
            g_loss_recon = self.SCTFLoss.reconstruct_loss(fake_images, Igt)
            g_loss_percep, g_loss_style = self.SCTFLoss.perceptron_and_style_loss(Igt, fake_images)
            g_loss_triple = self.SCTFLoss.triplet_loss(qkv_list)

            g_loss = (
                g_loss_fake * self.args.weight_g_fake 
                + g_loss_recon * self.args.weight_g_recon
                + g_loss_percep * self.args.weight_g_perceptual
                + g_loss_style * self.args.weight_g_style
                + g_loss_triple * self.args.weight_g_triplet
            )
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # ---------------------------------------
            #             3. Logging
            # ---------------------------------------
            loss_dict = {
                'D_loss' : d_loss,
                'D/loss_real' : d_loss_real.item() * self.args.weight_d_real,
                'D/loss_fake' : d_loss_fake.item() * self.args.weight_d_fake,
                'G_loss' : g_loss,
                'G/loss_fake' : g_loss_fake.item() * self.args.weight_g_fake,
                'G/loss_recon' : g_loss_recon.item() * self.args.weight_g_recon,
                'G/loss_style' : g_loss_style.item() * self.args.weight_g_style,
                'G/loss_percep' : g_loss_percep.item() * self.args.weight_g_perceptual,
                'G/loss_triple' : g_loss_triple.item() * self.args.weight_g_triplet
            }
            
            if (i + 1) % self.args.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "\nEpoch [{}/{}] | Elapsed [{}] | Iteration [{}/{}]\n".format(
                    epoch+1, self.args.epochs, et, i + 1, iterations
                )
                for tag, value in loss_dict.items():
                    log += "{}: {:.4f} ".format(tag, value)
                print(log)

    def train(self):
        torch.backends.cudnn.benchmark = True

        start_epoch = self.load_model()
        self.G.to(self.args.cuda)
        self.D.to(self.args.cuda)
        for epoch in range(start_epoch, self.args.epochs):
            print(f'epoch={epoch} training is start !')
            if epoch > 100:
                self.linear_decay_lr(epoch)

            self.train_one_epoch(epoch)
            self.save_model(epoch)
            
        print('training is finished...')
