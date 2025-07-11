import os, sys
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
from util import util
from PIL import Image 
import torch

# options
opt = TestOptions().parse()
opt.num_threads = 0 
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

# test stage

for i, data in enumerate(islice(dataset, opt.num_test)):
    # ------------------------------------------------------------
    # Generate fake IHC for the current pair
    # ------------------------------------------------------------
    with torch.no_grad():
        model.set_input(data)
        _, fake_B, _ = model.test()          # (1,C,H,W)

    # ------------------------------------------------------------
    # Build output filename  (same as ground-truth)
    # ------------------------------------------------------------
    fname   = os.path.basename(data['B_paths'][0])      # 00327.png
    out_dir = os.path.join(opt.results_dir, opt.name,
                           opt.phase + '_latest', 'images')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    # ------------------------------------------------------------
    # Save image
    # ------------------------------------------------------------
    fake_np = util.tensor2im(fake_B)         # or fake_B[0]
    Image.fromarray(fake_np).save(out_path)

    if opt.verbose:
        print(f"[{i+1}/{opt.num_test}] saved {out_path}")




#webpage.save()
