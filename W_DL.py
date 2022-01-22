## For 3D - 4 layers with reflection



############################ Import libs: ##########################
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import os
import numpy as np
from models import *

import torch
import torch.optim
from skimage.metrics import peak_signal_noise_ratio
from utils.denoising_utils import *
import time

t = time.localtime()
current_time = time.strftime("%m-%d %H:%M:%S", t)
print("current time = ", current_time, flush = True)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor      # Change to dtype = torch.FloatTensor for CPU

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.





############## LOAD IMAGE: ############################

import pandas as pd




# f = 'W_MLE.csv'
# dat_iter = np.asarray(pd.read_csv(f, header=None))  ## np.asarray gives different format
# dat_2 = dat_iter.reshape(20, 256, 256, 3)
# dat_2 = dat_2.transpose(1, 2, 0, 3)

# num_iter = 15000

def DL_W(W, reshape_vec, transpose_vec, n_x, n_y, n_z, num_iter):

	dat_iter = W
	dat_2 = dat_iter.reshape(n_z, n_x, n_y, 3)
	dat_2 = dat_2.transpose(1, 2, 0, 3)
	W_out = np.copy(W)


	for target in range(0, 3):
		print("\n\ntarget image/settings no:", target)

		dat_target = dat_2[:,:,:,target]
		scaling_factor = np.amax(dat_target)
		dat_target = dat_target*255/scaling_factor   ## Check this

		ar = np.clip(dat_target,0,255).astype(np.uint8)
		if dat_target.shape[0] == 1:
			ar = ar[0]
		else:
			ar = ar.transpose(0, 1, 2)
		act_image = ar


		## Pad images:
		new_shape = (act_image.shape[0] - act_image.shape[0] % 32, act_image.shape[1] - act_image.shape[1] % 32, 
					act_image.shape[2] - act_image.shape[2] % 32)
		if act_image.shape[0] % 32 != 0:
			tmp_1 = new_shape[0]+32
		else:
			tmp_1 = new_shape[0]
		if act_image.shape[1] % 32 != 0:
			tmp_2 = new_shape[1]+32
		else:
			tmp_2 = new_shape[1]
		if act_image.shape[2] % 32 != 0:
			tmp_3 = new_shape[2]+32
		else:
			tmp_3 = new_shape[2]
		new_shape = (tmp_1, tmp_2, tmp_3)
		print("padded shape: ", new_shape)

		img_noisy_pil = np.zeros(new_shape)
		img_noisy_pil[0:n_x,0:n_y,0:n_z] = act_image

		img_noisy_np = pil_to_np(img_noisy_pil)
		img_noisy_np = img_noisy_np[None, :]	## Added

		# As we don't have ground truth
		img_pil = img_noisy_pil
		img_np = img_noisy_np


		############### SETUP: #######################
		INPUT = 'noise' # 'meshgrid'
		pad = 'reflection'
		pad = 'zero'    ## WHY ITS ZERO???????


		OPT_OVER = 'net' # 'net,input'

		reg_noise_std = 1./30. # set to 1./20. for sigma=50
		LR = 0.01

		OPTIMIZER='adam' # 'LBFGS'
		show_every = 100
		exp_weight=0.99


		# num_iter = 3500
		# num_iter = 6500  # Subrata
		# num_iter = 15000  # Subrata new
		input_depth = 16
		figsize = 5

		net = skip(
				input_depth, 1, 
				num_channels_down = [16, 32, 64, 128],
				num_channels_up   = [16, 32, 64, 128],
				num_channels_skip = [0, 2, 4, 4],
				#upsample_mode='trilinear',
				upsample_mode='nearest',
				need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

		net = net.type(dtype)
		net_input = get_noise(input_depth, INPUT, (img_pil.shape[2], img_pil.shape[1], img_pil.shape[0])).type(dtype).detach()

		print("(img_pil.shape[2], img_pil.shape[1], img_pil.shape[0])", (img_pil.shape[2], img_pil.shape[1], img_pil.shape[0]))
		print("net_input.shape ", net_input.shape)
		
		
		# Compute number of parameters
		s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
		print ('Number of params: %d' % s)

		# Loss
		mse = torch.nn.MSELoss().type(dtype)
		img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
		print("img_noisy_torch.shape ", img_noisy_torch.shape)





		################# OPTIMIZE ######################
		global i, out_avg, psrn_noisy_last, last_net
		## Added by Subrata, 
		# See https://stackoverflow.com/questions/423379/using-global-variables-in-a-function
		# As it is a nested function definition, the out_avg is not in the original global scope
		# https://stackoverflow.com/questions/51662467/using-a-global-variable-inside-a-function-nested-in-a-function-in-python
		# somehow the net_input creates problem. 
		# https://stackoverflow.com/questions/5218895/python-nested-functions-variable-scoping  -- uses nonlocal


		net_input_saved = net_input.detach().clone()
		noise = net_input.detach().clone()
		out_avg = None
		last_net = None
		psrn_noisy_last = 0


		i = 0
		def closure():
			
			global i, out_avg, psrn_noisy_last, last_net, net_input

			if reg_noise_std > 0:
				net_input = net_input_saved + (noise.normal_() * reg_noise_std)
			
			out = net(net_input)
			
			# Smoothing
			if out_avg is None:
				out_avg = out.detach()
			else:
				out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
			
			total_loss = mse(out, img_noisy_torch)
			
			
			# torch.cuda.empty_cache()
			total_loss.backward()    ## Not happening with CUDA
			
			psrn_noisy = peak_signal_noise_ratio(img_noisy_np, out.detach().cpu().numpy()[0]) 
			psrn_gt    = peak_signal_noise_ratio(img_np, out.detach().cpu().numpy()[0]) 
			psrn_gt_sm = peak_signal_noise_ratio(img_np, out_avg.detach().cpu().numpy()[0]) 
			
			# Note that we do not have GT for the "snail" example
			# So 'PSRN_gt', 'PSNR_gt_sm' make no sense
			if i % 20 == 0:
				print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\n', flush=True)
			if  PLOT and i % show_every == 0:
				out_np = torch_to_np(out)
				out_np = out_np.transpose(0,3,2,1)
				out_np = out_np[0,0:n_x,0:n_y,0:n_z]
				out_np = out_np*scaling_factor/255
				# out_np = out_np.reshape(n_x* n_y * n_z)	
				# pd.DataFrame().to_csv("3D_"+str(target)+".csv", header=None, index=None)
				print(out_np.shape)
				#plt.imshow(out_np[:,:,10])
				#plt.savefig("images/3D_"+str(i)+"_test_4.pdf")
				#plot_image_grid([np.clip(out_np, 0, 1), 
			    #                 np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1, name="original_paper_images/steps"+str(i)+".pdf")
			if i % 1000 == 0:
				out_np = torch_to_np(out)
				out_np = out_np.transpose(0,3,2,1)
				out_np = out_np[0,0:n_x,0:n_y,0:n_z]
				out_np = out_np*scaling_factor
				
				pd.DataFrame(out_np.reshape(n_x * n_y * n_z)).to_csv("intermed/intermed/3D_"+str(target)+"_i_"+str(i)+"_test_4.csv", header=None, index=None)
			
			
			# Backtracking
			if i % show_every:
				if psrn_noisy - psrn_noisy_last < -5: 
					print('Falling back to previous checkpoint.')

					for new_param, net_param in zip(last_net, net.parameters()):
						net_param.data.copy_(new_param.cuda())

					return total_loss*0
				else:
					last_net = [x.detach().cpu() for x in net.parameters()]
					psrn_noisy_last = psrn_noisy
					
			i += 1

			return total_loss


		p = get_params(OPT_OVER, net, net_input)
		optimize(OPTIMIZER, p, closure, LR, num_iter)



		###############################################
		t = time.localtime()
		current_time = time.strftime("%m-%d %H:%M:%S", t)
		print("tgt: ", target, ", current time = ", current_time, flush = True)

		out_np = torch_to_np(net(net_input))

		print("shape is:", out_np.shape)
		out_np = out_np.transpose(0,3,2,1)
		print("shape is:", out_np.shape)
		out_np = out_np[0, 0:n_x, 0:n_y, 0:n_z]
		print("shape is:", out_np.shape)
		out_np = out_np*scaling_factor   ## /255


		#print("\ndat_target_orig: ", pd.DataFrame(dat_target_orig.reshape(n_x * n_y * n_z)).describe(), "\ndat_target: ", pd.DataFrame(dat_target.reshape(256* 256*20)).describe(), "\nout_np: ", pd.DataFrame(out_np.reshape(256* 256*20)).describe())

		plt.imshow(out_np[:,:,10])
		plt.savefig("intermed/3D_"+str(target)+"_test_4.pdf")
		out_np_rotated = out_np.transpose(2, 1, 0)
		pd.DataFrame(out_np.reshape(n_x* n_y * n_z)).to_csv("intermed/3D_"+str(target)+"_test_4.csv", header=None, index=None)
		
		# add transpose 

		W_out[target] = out_np_rotated.reshape(n_x * n_y * n_z)
		time.sleep(50)

	return W_out


