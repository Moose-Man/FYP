# FYP
Predicting IHC Images from H&;E Images for Breast Cancer

## Model Performance

| Model               | With PatchNCE | With STN | PSNR   | SSIM    |
|---------------------|:-------------:|:--------:|:------:|:-------:|
| **pyramidPix2Pix**  |               |          |  19.24  | 0.3825 |
|                     |               |    ✓     | 19.98  | 0.4299  |
|                     |      ✓        |          |  20.15 | 0.3870  |
|                     |      ✓        |    ✓     |  19.52 | 0.4259 |
| **BiCycleGAN**      |               |          | 16.27  | 0.3249  |
|                     |               |    ✓     | 16.54  | 0.3388  |
|                     |      ✓        |          |        |         |
|                     |      ✓        |    ✓     |        |         |
| **PAN**             |               |          |  21.50  | 0.3944  |
|                     |               |    ✓     |  20.23  | 0.3827  |
|                     |      ✓        |          |        |         |
|                     |      ✓        |    ✓     |        |         |

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

Reduced Epoch Count
• Training epochs were cut from 200 to 30.

Reduced Network Widths
• Base channel counts ngf / ndf / nef lowered from 64 to 32 to fit laptop GPU memory while keeping the U-Net-256 architecture unchanged.

Upsample-to-256 Training Resolution
• Although raw dataset tiles are 128 × 128, flags --load_size 286 --crop_size 256 upscale every H&E ↔ IHC pair to 286 × 286 then random cropping to 256x256 before entering the network, so both the baseline and STN models learn and generate at 256-pixel resolution (this was unintentional, and the flags were both meant to have values 128. It is not a destructive error, however, so the results are still meaningful)

Commmand used to train:
```
python train.py --dataroot "C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset" --dataset_mode aligned --model bicycle_gan --name he2ihc_bicycle_baseline --input_nc 3 --output_nc 3 --load_size 286 --crop_size 256 --ngf 32 --ndf 32 --nef 32 --batch_size 2 --niter 20 --niter_decay 10 --display_id -1 --print_freq 100 --display_freq 400 --update_html_freq 400 --save_epoch_freq 5
```
Command used to generate images (test.py has been altered to only provide a single sample output image of matching filenames with the test dataset)
```
python test.py --dataroot "C:\Users\user\Desktop\Uni_work\year_3\FYP\code\Pyramid_Pix2Pix\BCI_dataset" --dataset_mode aligned --model bicycle_gan --name he2ihc_bicycle_baseline --input_nc 3 --output_nc 3 --phase test --serial_batches --num_test 977 --epoch latest --ngf 32 --ndf 32 --nef 32 --n_samples 1
```

### adjustments: PAN

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



