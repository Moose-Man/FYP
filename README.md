# FYP
Predicting IHC Images from H&;E Images for Breast Cancer

## Model Performance

| Model               | With PatchNCE | With STN | PSNR   | SSIM    |
|---------------------|:-------------:|:--------:|:------:|:-------:|
| **pyramidPix2Pix**  |               |          |  19.24  | 0.3825 |
|                     |               |    ✓     | 19.98  | 0.4299  |
|                     |      ✓        |          |  20.15 | 0.3870  |
|                     |      ✓        |    ✓     |  19.52 | 0.4259 |
| **BiCycleGAN**      |               |          |  16.92  | 0.3502 |
|                     |               |    ✓     |  17.56 |  0.3652 |
| **PAN**             |               |          |  19.63 |  0.3359  |
|                     |               |    ✓     |  18.75 |  0.3452 |

### adjustments: pyramidPix2Pix

Model has been edited to be purely deterministic for testing purposes (see pyramid_pix2pix_ver12.py)

Reduced Epoch Count & Adjusted Learning Rate
• Training epochs were cut from 100 down to 50.
• The optimizer’s learning rate was correspondingly lowered to maintain stable convergence at fewer epochs. 

Input Resolution Downscaling
• All HE–IHC patches were resized directly to 128×128 pixels

Online Gaussian Pyramid Generation
• The model now builds the multi-scale pyramid on the fly from IHC outputs.

Dataset Normalization
• Added per-channel mean-zero, unit-variance normalization before feeding images into both generator and discriminator.

PatchNCE Encoder Downsizing
• To avoid out-of-memory errors when sampling negative patches, the PatchNCE module’s feature maps were kept at a lower dimensionality (256 channels) and negative sample count was reduced.

### adjustments: BiCycleGAN

fixed seed initialization
during training:
```
# Ensures that even inherently nondeterministic CUDA/CUDNN algorithms (convolutions, grid-sample, matrix multiplies, etc.) behave deterministically
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(opt.seed) # fixes any use of python's built in RNG
np.random.seed(opt.seed) # fixes any use np.python's built in RNG
torch.manual_seed(opt.seed) # fixes all CPU side RNG
torch.cuda.manual_seed_all(opt.seed) # fixes all GPU side RNG
```
during testing:
```
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
```

Reduced Epoch Count
• Training epochs were cut from 200 to 30.

Reduced Network Widths
• Base channel counts ngf / ndf / nef lowered from 64 to 32 to fit laptop GPU memory while keeping the U-Net-256 architecture unchanged.

Upsample-to-256 Training Resolution
• Although raw dataset tiles are 128 × 128, flags --load_size 286 --crop_size 256 upscale every H&E ↔ IHC pair to 286 × 286 then random cropping to 256x256 before entering the network, so both the baseline and STN models learn and generate at 256-pixel resolution (this was unintentional, and the flags were both meant to have values 128. It is not a destructive error, however, so the results are still meaningful)

Commmand used to train:
```
python train.py --dataroot "C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset" --dataset_mode aligned --model bicycle_gan --name he2ihc_bicycle_stn+patchnce --input_nc 3 --output_nc 3 --load_size 286 --crop_size 256 --ngf 32 --ndf 32 --nef 32 --batch_size 36 --niter 20 --niter_decay 10 --display_id -1 --print_freq 100 --display_freq 400 --update_html_freq 400 --save_epoch_freq 5 --lambda_stn 4 --lambda_nce 0.55 --temperature_nce 0.085 --num_negatives_nce 128 --gpu_ids 0

#omit lines 'lambda_nce 0.55 --temperature_nce 0.085 --num_negatives_nce 128' to disable patchnce
#omit line 'lambda_stn 4' to disable stn
#ensure model names are changed if training multiple variations to avoid checkpoint overwrites
```
Command used to generate images (test.py has been altered to only provide a single sample output image of matching filenames with the test dataset)
```
python test.py --dataroot "C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset" --dataset_mode aligned --model bicycle_gan --name he2ihc_bicycle_baseline --input_nc 3 --output_nc 3 --phase test --serial_batches --num_test 977 --epoch latest --ngf 32 --ndf 32 --nef 32 --n_samples 1
```

### adjustments: PAN

fixed seed initialization
during training:
```
random.seed(opt.seed) # fixes any use of python's built in RNG
np.random.seed(opt.seed) # fixes any use np.python's built in RNG
torch.manual_seed(opt.seed) # fixes all CPU side RNG
torch.cuda.manual_seed_all(opt.seed) # fixes all GPU side RNG

# Ensures that even inherently nondeterministic CUDA/CUDNN algorithms (convolutions, grid-sample, matrix multiplies, etc.) behave deterministically
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
during testing:
```
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
```

Training schedule
• 20 epochs + 10 epoch LR-decay (--niter 20 --niter_decay 10)

Batch & resolution
• Batch = 2 (vs 4) • Patch = 128² (--loadSize 128 --fineSize 128)

Generator / Discriminator widths
• --which_model_netG unet_128 (64→128 U-Net depth) 
• ndf/ngf kept at 64 (defaults)

GAN loss
• Kept default LSGAN (--no_lsgan omitted) to avoid BCE‐logit crash on new PyTorch.

Weight initialisation
• Restored helper init_net() and added flag --init_gain 0.02 (default).

STN branch 
• Added STN() module; enable with --lambda_stn 4 (value > 0 activates STN module). 
• STN is initialised & moved to GPU inside pan_model.py.

Windows stability
• Forced single-process dataloader (--nThreads 0).

Deprecation fixes
• Replaced deprecated np.float, .cuda(device_id=…), and old weight-init calls.

Test-time convenience
• Custom script test_singleoutput.py
– inserts repo path into sys.path
– loads netG only
– saves one fake_B per input, same filename+ext
– works with dataset_mode aligned and --nThreads 0.

command used to train:
```
python train.py --dataroot "C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\PANdataset" --dataset_mode aligned --model pan --name pan_stn --which_direction AtoB --which_model_netG unet_128 --input_nc 3 --output_nc 3 --loadSize 128 --fineSize 128 --batchSize 2 --nThreads 0 --niter 20 --niter_decay 10 --pan_lambdas 5 1 1 1 5 --print_freq 100 --save_epoch_freq 5 --display_id -1 --lambda_stn 4 --init_gain 0.02
```

command used to test:
```
python test_singleoutput.py --dataroot C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset\PANdataset --dataset_mode aligned --name pan_baseline --model pan --which_direction AtoB --which_model_netG unet_128 --phase test --which_epoch latest --loadSize 128 --fineSize 128 --results_dir results_single --nThreads 0
```



