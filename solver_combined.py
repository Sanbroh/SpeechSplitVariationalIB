# from model import Generator_3 as Generator
# from model import Generator_6 as F0_Converter  # Import Generator_6 for pitch conversion
# from model import InterpLnr
# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import time
# import datetime
# import pickle

# from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy

# # use demo data for simplicity; you can use your own validation set as needed
# validation_pt = pickle.load(open('assets/demo.pkl', "rb"))

# class Solver(object):
#     """Solver for training"""

#     def __init__(self, vcc_loader, config, hparams):
#         """Initialize configurations."""
#         self.vcc_loader = vcc_loader
#         self.hparams = hparams

#         # Training configurations.
#         self.num_iters = config.num_iters
#         self.g_lr = config.g_lr
#         self.beta1 = config.beta1
#         self.beta2 = config.beta2
#         self.resume_iters = config.resume_iters

#         # Miscellaneous.
#         self.use_tensorboard = config.use_tensorboard
#         self.use_cuda = torch.cuda.is_available()
#         self.device = torch.device('cuda:{}'.format(config.device_id) if self.use_cuda else 'cpu')

#         # Directories.
#         self.log_dir = config.log_dir
#         self.sample_dir = config.sample_dir
#         self.model_save_dir = config.model_save_dir

#         # Step size.
#         self.log_step = config.log_step
#         self.sample_step = config.sample_step
#         self.model_save_step = config.model_save_step

#         # Build models and (optionally) tensorboard.
#         self.build_model()
#         if self.use_tensorboard:
#             self.build_tensorboard()

#     def build_model(self):
#         # Main speech conversion generator.
#         self.G = Generator(self.hparams)
#         # F0 conversion generator (mini SPEECHSPLIT variant without content encoder).
#         self.P = F0_Converter(self.hparams)
#         # Interpolator for random segment sampling.
#         self.Interp = InterpLnr(self.hparams)

#         # Create separate optimizers.
#         self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
#         self.p_optimizer = torch.optim.Adam(self.P.parameters(), self.g_lr, [self.beta1, self.beta2])

#         self.print_network(self.G, 'G')
#         self.print_network(self.P, 'P')

#         self.G.to(self.device)
#         self.P.to(self.device)
#         self.Interp.to(self.device)

#     def print_network(self, model, name):
#         num_params = sum(p.numel() for p in model.parameters())
#         print(model)
#         print(name)
#         print("The number of parameters: {}".format(num_params))

#     def print_optimizer(self, opt, name):
#         print(opt)
#         print(name)

#     def restore_model(self, resume_iters):
#         print('Loading the trained models from step {}...'.format(resume_iters))
#         G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
#         p_path = os.path.join(self.model_save_dir, '{}-P.ckpt'.format(resume_iters))
#         g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
#         p_checkpoint = torch.load(p_path, map_location=lambda storage, loc: storage)
#         self.G.load_state_dict(g_checkpoint['model'])
#         self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
#         self.P.load_state_dict(p_checkpoint['model'])
#         self.p_optimizer.load_state_dict(p_checkpoint['optimizer'])
#         self.g_lr = self.g_optimizer.param_groups[0]['lr']

#     def build_tensorboard(self):
#         from torch.utils.tensorboard import SummaryWriter
#         self.writer = SummaryWriter(self.log_dir)

#     def reset_grad(self):
#         self.g_optimizer.zero_grad()
#         self.p_optimizer.zero_grad()

#     def train(self):
#         data_loader = self.vcc_loader
#         data_iter = iter(data_loader)
#         start_iters = 0
#         if self.resume_iters:
#             print('Resuming ...')
#             start_iters = self.resume_iters
#             self.num_iters += self.resume_iters
#             self.restore_model(self.resume_iters)
#             self.print_optimizer(self.g_optimizer, 'G_optimizer')
#             self.print_optimizer(self.p_optimizer, 'P_optimizer')

#         g_lr = self.g_lr
#         print('Current learning rates, g_lr: {}.'.format(g_lr))
#         keys = ['G/loss_id', 'P/loss_f0']
#         print('Start training...')
#         start_time = time.time()

#         # Initialize lists to record losses for plotting.
#         iter_list = []
#         loss_g_list = []
#         loss_p_list = []

#         val_iters = []
#         val_losses_G = []
#         val_losses_P = []

#         for i in range(start_iters, self.num_iters):
#             try:
#                 x_real_org, emb_org, f0_org, len_org = next(data_iter)
#             except:
#                 data_iter = iter(data_loader)
#                 x_real_org, emb_org, f0_org, len_org = next(data_iter)
            
#             x_real_org = x_real_org.to(self.device)
#             emb_org = emb_org.to(self.device)
#             len_org = len_org.to(self.device)
#             f0_org = f0_org.to(self.device)

#             # ----------------------------- Train Generator (G) -----------------------------
#             self.G.train()
#             # Form the input by concatenating speech and its pitch (for joint rhythm conversion)
#             x_f0 = torch.cat((x_real_org, f0_org), dim=-1)
#             x_f0_intrp = self.Interp(x_f0, len_org)
#             f0_org_intrp, _ = quantize_f0_torch(x_f0_intrp[:, :, -1])
#             x_f0_intrp_org = torch.cat((x_f0_intrp[:, :, :-1], f0_org_intrp), dim=-1)
#             x_identic = self.G(x_f0_intrp_org, x_real_org, emb_org)
#             g_loss_id = F.mse_loss(x_real_org, x_identic, reduction='mean')

#             # -------------------------- Train Pitch Converter (P) --------------------------
#             self.P.train()
#             # For pitch conversion, we use the original speech input and the pitch contour.
#             # We assume f0_org is preprocessed and can be quantized.
#             # Here, we compute the output from P and apply cross-entropy loss.
#             # First, get one-hot target for pitch conversion.
#             f0_target_onehot, f0_target_labels = quantize_f0_torch(f0_org)
#             # f0_target_labels will be used for cross-entropy loss (as class indices).
#             f0_output = self.P(x_real_org, f0_target_onehot)  # Output shape: (B, T, 257)
#             loss_f0 = F.cross_entropy(f0_output.view(-1, 257), f0_target_labels.view(-1), reduction='mean')

#             # Total loss is the sum (or weighted sum) of the two losses.
#             total_loss = g_loss_id + loss_f0  # You may introduce a weight lambda for loss_f0 if desired.

#             # Backpropagate both optimizers.
#             self.reset_grad()
#             total_loss.backward()
#             self.g_optimizer.step()
#             self.p_optimizer.step()

#             loss = {}
#             loss['G/loss_id'] = g_loss_id.item()
#             loss['P/loss_f0'] = loss_f0.item()

#             # ----------------------------- Logging and Checkpoints -----------------------------
#             # Record losses at each logging step.
#             if (i+1) % self.log_step == 0:
#                 iter_list.append(i+1)
#                 loss_g_list.append(loss['G/loss_id'])
#                 loss_p_list.append(loss['P/loss_f0'])

#                 et = time.time() - start_time
#                 et = str(datetime.timedelta(seconds=et))[:-7]
#                 log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
#                 for tag in keys:
#                     log += ", {}: {:.8f}".format(tag, loss[tag])
#                 print(log)
#                 if self.use_tensorboard:
#                     for tag, value in loss.items():
#                         self.writer.add_scalar(tag, value, i+1)

#             if (i+1) % self.model_save_step == 0:
#                 G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
#                 P_path = os.path.join(self.model_save_dir, '{}-P.ckpt'.format(i+1))
#                 torch.save({'model': self.G.state_dict(),
#                             'optimizer': self.g_optimizer.state_dict()}, G_path)
#                 torch.save({'model': self.P.state_dict(),
#                             'optimizer': self.p_optimizer.state_dict()}, P_path)
#                 print('Saved model checkpoints into {}...'.format(self.model_save_dir))

#             # ----------------------------- Validation -----------------------------
#             if (i+1) % self.sample_step == 0:
#                 self.G.eval()
#                 self.P.eval()
#                 with torch.no_grad():
#                     # ------------------------------
#                     # Validation for the G component
#                     # ------------------------------
#                     total_loss_G = 0.0
#                     num_samples_G = 0
#                     for demo_entry in validation_pt:
#                         # Unpack the demo entry:
#                         # demo_entry[0]: speaker_id
#                         # demo_entry[1]: speaker_embedding (numpy array)
#                         # demo_entry[2]: (mel_padded, f0_onehot, num_frames, utt_id)
#                         speaker_id = demo_entry[0]
#                         speaker_emb_np = demo_entry[1]
#                         mel_padded, f0_onehot_np, num_frames, utt_id = demo_entry[2]

#                         # Convert to tensors with a batch dimension.
#                         emb_tensor = torch.from_numpy(speaker_emb_np).unsqueeze(0).to(self.device)
#                         x_real_pad = torch.from_numpy(mel_padded).unsqueeze(0).to(self.device)   # shape: (1, T, n_mels)
#                         f0_tensor = torch.from_numpy(f0_onehot_np).unsqueeze(0).to(self.device)   # shape: (1, T, num_bins+1)

#                         # Form the combined input (x_f0) for G.
#                         x_f0_val = torch.cat([x_real_pad, f0_tensor], dim=-1)  # shape: (1, T, n_mels + num_bins+1)

#                         # Forward pass through G.
#                         reconstructed = self.G(x_f0_val, x_real_pad, emb_tensor)
#                         loss_G = F.mse_loss(x_real_pad, reconstructed, reduction='sum')
#                         total_loss_G += loss_G.item()
#                         num_samples_G += 1

#                     avg_loss_G = total_loss_G / num_samples_G
#                     print("Validation G loss: {:.4f}".format(avg_loss_G))
#                     if self.use_tensorboard:
#                         self.writer.add_scalar('Validation_loss_G', avg_loss_G, i+1)

#                     # ------------------------------
#                     # Validation for the P component
#                     # ------------------------------
#                     total_loss_P = 0.0
#                     num_samples_P = 0
#                     for demo_entry in validation_pt:
#                         # Unpack the demo entry.
#                         speaker_id = demo_entry[0]
#                         # Use the stored speaker embedding if needed.
#                         speaker_emb_np = demo_entry[1]
#                         mel_padded, f0_onehot_np, num_frames, utt_id = demo_entry[2]

#                         # Convert mel spectrogram (x_org) and pitch contour (f0_onehot) into tensors.
#                         x_org = torch.from_numpy(mel_padded).unsqueeze(0).to(self.device)    # (1, T, n_mels)
#                         f0_target_onehot = torch.from_numpy(f0_onehot_np).unsqueeze(0).to(self.device)  # (1, T, num_bins+1)

#                         # Derive target pitch labels from the one-hot representation.
#                         target_labels = f0_target_onehot.argmax(dim=-1)  # shape: (1, T)

#                         # Forward pass through the pitch converter P.
#                         # P expects two inputs: the original speech (x_org) and the pitch contour (f0_target_onehot).
#                         f0_output = self.P(x_org, f0_target_onehot)  # Expected shape: (1, T, 257)

#                         # Compute cross-entropy loss.
#                         loss_P = F.cross_entropy(f0_output.view(-1, 257), target_labels.view(-1), reduction='sum')
#                         total_loss_P += loss_P.item()
#                         num_samples_P += 1

#                     avg_loss_P = total_loss_P / num_samples_P
#                     print("Validation P loss: {:.4f}".format(avg_loss_P))
#                     if self.use_tensorboard:
#                         self.writer.add_scalar('Validation_loss_P', avg_loss_P, i+1)

#                     val_iters.append(i+1)
#                     val_losses_G.append(avg_loss_G)
#                     val_losses_P.append(avg_loss_P)
            
#                 # Set the models back to training mode for the next iteration.
#                 self.G.train()
#                 self.P.train()

#         loss_data = {
#             'iter_list': iter_list,
#             'loss_g_list': loss_g_list,
#             'loss_p_list': loss_p_list
#         }
#         loss_data_path = os.path.join(self.sample_dir, 'loss_data.pkl')
#         with open(loss_data_path, 'wb') as f:
#             pickle.dump(loss_data, f)
#         print("Loss data saved to {}".format(loss_data_path))

#         # Plot Speech Reconstruction Loss (G)
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(10,6))
#         plt.plot(iter_list, loss_g_list, label='Speech Reconstruction Loss (G)', color='blue')
#         plt.xlabel('Iteration')
#         plt.ylabel('Loss')
#         plt.title('Speech Reconstruction Loss Curve')
#         plt.legend()
#         if iter_list:
#             initial_g = loss_g_list[0]
#             final_g = loss_g_list[-1]
#             plt.text(iter_list[0], initial_g, f"Initial G: {initial_g:.4f}", fontsize=9, color='blue', verticalalignment='bottom')
#             plt.text(iter_list[-1], final_g, f"Final G: {final_g:.4f}", fontsize=9, color='blue', verticalalignment='top')
#         plot_path_G = os.path.join(self.sample_dir, 'loss_plot_G.png')
#         plt.savefig(plot_path_G, dpi=150)
#         plt.close()
#         print("Speech Reconstruction Loss plot saved to {}".format(plot_path_G))

#         # Plot Pitch Conversion Loss (P)
#         plt.figure(figsize=(10,6))
#         plt.plot(iter_list, loss_p_list, label='Pitch Conversion Loss (P)', color='orange')
#         plt.xlabel('Iteration')
#         plt.ylabel('Loss')
#         plt.title('Pitch Conversion Loss Curve')
#         plt.legend()
#         if iter_list:
#             initial_p = loss_p_list[0]
#             final_p = loss_p_list[-1]
#             plt.text(iter_list[0], initial_p, f"Initial P: {initial_p:.4f}", fontsize=9, color='orange', verticalalignment='top')
#             plt.text(iter_list[-1], final_p, f"Final P: {final_p:.4f}", fontsize=9, color='orange', verticalalignment='bottom')
#         plot_path_P = os.path.join(self.sample_dir, 'loss_plot_P.png')
#         plt.savefig(plot_path_P, dpi=150)
#         plt.close()
#         print("Pitch Conversion Loss plot saved to {}".format(plot_path_P))

#         # VALIDATION
#         validation_data = {
#             "iterations": val_iters,
#             "loss_G": val_losses_G,
#             "loss_P": val_losses_P
#         }
#         with open(os.path.join(self.model_save_dir, "validation_losses.pkl"), "wb") as f:
#             pickle.dump(validation_data, f)
#         print("Saved validation losses to validation_losses.pkl")

#         # Plot the validation losses.
#         plt.figure()
#         plt.plot(val_iters, val_losses_G, label="Validation Loss G")
#         plt.plot(val_iters, val_losses_P, label="Validation Loss P")
#         plt.xlabel("Iteration")
#         plt.ylabel("Loss")
#         plt.legend()
#         plt.title("Validation Losses over Iterations")
#         plt.savefig(os.path.join(self.model_save_dir, "validation_loss_plot.png"))
#         plt.close()
#         print("Saved validation loss plot to validation_loss_plot.png")

from model import Generator_3 as Generator
from cost_functions import get_KLdivergence_loss
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

# use demo data for simplicity; you can use your own validation set as needed
validation_pt = pickle.load(open('assets/demo_m2f.pkl', "rb"))

class Solver(object):
    """Solver for training"""

    def __init__(self, vcc_loader, config, hparams):
        """Initialize configurations."""
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

        # Build models and (optionally) tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        # Main speech conversion generator.
        self.G = Generator(self.hparams)
        # Interpolator for random segment sampling.
        self.Interp = InterpLnr(self.hparams)

        # Create optimizer for G.
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])

        self.print_network(self.G, 'G')

        self.G.to(self.device)
        self.Interp.to(self.device)

    def print_network(self, model, name):
        num_params = sum(p.numel() for p in model.parameters())
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def print_optimizer(self, opt, name):
        print(opt)
        print(name)

    def restore_model(self, resume_iters):
        print('Loading the trained model from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
        self.G.load_state_dict(g_checkpoint['model'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
        self.g_lr = self.g_optimizer.param_groups[0]['lr']

    def build_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def train(self):
        data_loader = self.vcc_loader
        data_iter = iter(data_loader)
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.g_optimizer, 'G_optimizer')

        g_lr = self.g_lr
        print('Current learning rates, g_lr: {}.'.format(g_lr))
        keys = ['G/loss_id']
        print('Start training...')
        start_time = time.time()

        # Initialize lists to record losses for plotting.
        iter_list = []
        loss_g_list = []

        val_iters = []
        val_losses_G = []

        for i in range(start_iters, self.num_iters):
            try:
                x_real_org, emb_org, f0_org, len_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_org, emb_org, f0_org, len_org = next(data_iter)
            
            x_real_org = x_real_org.to(self.device)
            emb_org = emb_org.to(self.device)
            len_org = len_org.to(self.device)
            f0_org = f0_org.to(self.device)

            # ----------------------------- Train Generator (G) -----------------------------
            self.G.train()
            # Form the input by concatenating speech and its pitch (for joint rhythm conversion)
            x_f0 = torch.cat((x_real_org, f0_org), dim=-1)
            x_f0_intrp = self.Interp(x_f0, len_org)
            f0_org_intrp, _ = quantize_f0_torch(x_f0_intrp[:, :, -1])
            x_f0_intrp_org = torch.cat((x_f0_intrp[:, :, :-1], f0_org_intrp), dim=-1)
            x_identic, var, mu = self.G(x_f0_intrp_org, x_real_org, emb_org)
            # g_loss_id = F.mse_loss(x_real_org, x_identic, reduction='mean')
            g_loss_id = get_KLdivergence_loss(x_real_org, x_identic, mu, var, beta=50, r_method='mean')

            # Total loss is now just the reconstruction loss.
            total_loss = g_loss_id

            # Backpropagation for G.
            self.reset_grad()
            total_loss.backward()
            self.g_optimizer.step()

            loss = {}
            loss['G/loss_id'] = g_loss_id.item()

            # ----------------------------- Logging and Checkpoints -----------------------------
            if (i+1) % self.log_step == 0:
                iter_list.append(i+1)
                loss_g_list.append(loss['G/loss_id'])

                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.8f}".format(tag, loss[tag])
                print(log)
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.writer.add_scalar(tag, value, i+1)

            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                torch.save({'model': self.G.state_dict(),
                            'optimizer': self.g_optimizer.state_dict()}, G_path)
                print('Saved model checkpoint for G into {}...'.format(self.model_save_dir))

            # ------------------------- Validation using new demo.pkl structure -------------------------
            if (i+1) % self.sample_step == 0:
                self.G.eval()
                with torch.no_grad():
                    loss_val = []
                    for val_sub in validation_pt:
                        # Unpack the demo entry:
                        #   val_sub[0]: speaker ID (e.g., "p558")
                        #   val_sub[1]: speaker embedding (a 1D numpy array)
                        #   val_sub[2]: a tuple (spect, f0, num_frames, utt_id)
                        speaker_id = val_sub[0]
                        spk_emb = val_sub[1]
                        spect, f0, num_frames, utt_id = val_sub[2]

                        # Convert speaker embedding to tensor and ensure batch dimension.
                        emb_org_val = torch.from_numpy(spk_emb).to(self.device)
                        if emb_org_val.dim() == 1:
                            emb_org_val = emb_org_val.unsqueeze(0)

                        # Pad the mel spectrogram to fixed length (e.g., 192 frames).
                        x_real_pad, _ = pad_seq_to_2(spect[np.newaxis, :, :], 192)
                        x_real_pad = torch.from_numpy(x_real_pad).to(self.device)

                        # Create a tensor for the length of the utterance.
                        len_org = torch.tensor([num_frames]).to(self.device)

                        # Process the F0 data:
                        # If f0 is 1D (raw contour), pad it to length 192.
                        if f0.ndim == 1:
                            f0_pad = np.pad(f0, (0, 192 - num_frames), 'constant', constant_values=(0, 0))
                            f0_quantized = quantize_f0_numpy(f0_pad)[0]
                        else:
                            # If f0 is already 2D, adjust its length if necessary.
                            if f0.shape[0] < 192:
                                f0_quantized = np.pad(f0, ((0, 192 - f0.shape[0]), (0, 0)),
                                                    'constant', constant_values=(0, 0))
                            else:
                                f0_quantized = f0[:192, :]
                        # Add batch dimension.
                        f0_onehot = f0_quantized[np.newaxis, :, :]
                        f0_org_val = torch.from_numpy(f0_onehot).to(self.device)

                        # Create combined input by concatenating the mel spectrogram and F0 along the feature dimension.
                        x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)

                        # Forward pass through the generator.
                        x_identic_val, _, _ = self.G(x_f0, x_real_pad, emb_org_val)

                        # Compute reconstruction loss (MSE) between the original mel and the generated output.
                        g_loss_val = F.mse_loss(x_real_pad, x_identic_val, reduction='sum')
                        loss_val.append(g_loss_val.item())

                    val_loss = np.mean(loss_val)
                    print('Validation loss: {}'.format(val_loss))
                    if self.use_tensorboard:
                        self.writer.add_scalar('Validation_loss', val_loss, i+1)

                    val_iters.append(i+1)
                    val_losses_G.append(val_loss)
                self.G.train()

        loss_data = {
            'iter_list': iter_list,
            'loss_g_list': loss_g_list
        }
        loss_data_path = os.path.join(self.sample_dir, 'loss_data.pkl')
        with open(loss_data_path, 'wb') as f:
            pickle.dump(loss_data, f)
        print("Loss data saved to {}".format(loss_data_path))

        # Plot Speech Reconstruction Loss (G)
        plt.figure(figsize=(10,6))
        plt.plot(iter_list, loss_g_list, label='Speech Reconstruction Loss (G)', color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Speech Reconstruction Loss Curve')
        plt.legend()
        if iter_list:
            initial_g = loss_g_list[0]
            final_g = loss_g_list[-1]
            plt.text(iter_list[0], initial_g, f"Initial G: {initial_g:.4f}", fontsize=9, color='blue', verticalalignment='bottom')
            plt.text(iter_list[-1], final_g, f"Final G: {final_g:.4f}", fontsize=9, color='blue', verticalalignment='top')
        plot_path_G = os.path.join(self.sample_dir, 'loss_plot_G.png')
        plt.savefig(plot_path_G, dpi=150)
        plt.close()
        print("Speech Reconstruction Loss plot saved to {}".format(plot_path_G))

        # VALIDATION: Save and plot validation losses for G only.
        validation_data = {
            "iterations": val_iters,
            "loss_G": val_losses_G
        }
        with open(os.path.join(self.sample_dir, "validation_losses.pkl"), "wb") as f:
            pickle.dump(validation_data, f)
        print("Saved validation losses to validation_losses.pkl")

        plt.figure()
        plt.plot(val_iters, val_losses_G, label="Validation Loss G", color='blue')
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Validation Loss over Iterations")
        plt.savefig(os.path.join(self.sample_dir, "validation_loss_plot.png"))
        plt.close()
        print("Saved validation loss plot to validation_loss_plot.png")
