from model import Generator_3 as Generator
from model import InterpLnr
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle

from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy

# use demo data for simplicity
# make your own validation set as needed
validation_pt = pickle.load(open('assets/demo.pkl', "rb"))

class Solver(object):
    """Solver for training"""

    def __init__(self, vcc_loader, config, hparams):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.hparams = hparams

        # Training configurations.
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        
        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

            
    def build_model(self):        
        self.G = Generator(self.hparams)
        
        self.Interp = InterpLnr(self.hparams)
            
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        
        self.G.to(self.device)
        self.Interp.to(self.device)

        
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        
        
    def print_optimizer(self, opt, name):
        print(opt)
        print(name)
        
        
    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
        self.G.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']
        
        
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
#=====================================================================================================================
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.g_optimizer, 'G_optimizer')
                        
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        print ('Current learning rates, g_lr: {}.'.format(g_lr))
        
        # Print logs in specified order
        keys = ['G/loss_id']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real_org, emb_org, f0_org, len_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_org, emb_org, f0_org, len_org = next(data_iter)
            
            x_real_org = x_real_org.to(self.device)
            emb_org = emb_org.to(self.device)
            len_org = len_org.to(self.device)
            f0_org = f0_org.to(self.device)
            
                    
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_f0 = torch.cat((x_real_org, f0_org), dim=-1)
            x_f0_intrp = self.Interp(x_f0, len_org) 
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:,:,-1])[0]
            x_f0_intrp_org = torch.cat((x_f0_intrp[:,:,:-1], f0_org_intrp), dim=-1)
            
            x_identic = self.G(x_f0_intrp_org, x_real_org, emb_org)
            g_loss_id = F.mse_loss(x_real_org, x_identic, reduction='mean') 
           
            # Backward and optimize.
            g_loss = g_loss_id
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.8f}".format(tag, loss[tag])
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.writer.add_scalar(tag, value, i+1)
                        
                        
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                torch.save({'model': self.G.state_dict(),
                            'optimizer': self.g_optimizer.state_dict()}, G_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))            
            

            # Validation.
            if (i+1) % self.sample_step == 0:
                self.G.eval()
                with torch.no_grad():
                    loss_val = []
                    # Here, validation_pt is assumed to be loaded from demo.pkl
                    for val_sub in validation_pt:
                        # Extract speaker embedding and convert to tensor.
                        emb_org_val = torch.from_numpy(val_sub[1]).unsqueeze(0).to(self.device)
                        
                        # Unpack the feature tuple.
                        # Each tuple is (mel_padded, f0_onehot, num_frames, utterance_id)
                        mel_spec, f0_onehot, num_frames, utt_id = val_sub[2]
                        
                        # Convert mel_spec and f0_onehot to torch tensors with a batch dimension.
                        # mel_spec shape: (192, n_mels); f0_onehot shape: (192, num_bins+1)
                        x_real_pad = torch.from_numpy(mel_spec).unsqueeze(0).to(self.device)
                        f0_org_val = torch.from_numpy(f0_onehot).unsqueeze(0).to(self.device)
                        
                        # Concatenate along the last dimension to form x_f0.
                        # x_f0 shape becomes: (1, 192, n_mels + num_bins+1)
                        x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
                        
                        # Forward pass through the generator.
                        # (Generator_3 returns mel_outputs, mu, var)
                        x_identic_val = self.G(x_f0, x_real_pad, emb_org_val)
                        
                        # Compute loss: here we use mean squared error between the original and reconstructed mel-spectrogram.
                        g_loss_val = F.mse_loss(x_real_pad, x_identic_val, reduction='sum')
                        loss_val.append(g_loss_val.item())
                        
                    val_loss = np.mean(loss_val)
                    print('Validation loss: {}'.format(val_loss))
                    if self.use_tensorboard:
                        self.writer.add_scalar('Validation_loss', val_loss, i+1)
                    

            # plot test samples
            # Removed
