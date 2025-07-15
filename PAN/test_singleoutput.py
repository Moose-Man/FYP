import os, sys, torch
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)          # ensure PAN packages are on path

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.util import tensor2im, save_image



opt = TestOptions().parse()
opt.serial_batches = True        # deterministic order
opt.no_html = True
data_loader = CreateDataLoader(opt)
dataset     = data_loader.load_data()
model       = create_model(opt)
model.netG.eval()

save_root = os.path.join(
    opt.results_dir, opt.name, f'{opt.phase}_{opt.which_epoch}')
os.makedirs(save_root, exist_ok=True)

for i, data in enumerate(dataset):
    model.set_input(data)
    model.test()
    
    vis = model.get_current_visuals()['fake_B']
    if isinstance(vis, torch.Tensor):            # old-style models
        fake_B = tensor2im(vis)
    else:                                        # already a NumPy array
        fake_B = vis

    # ───── keep original filename & extension ─────
    in_path   = model.get_image_paths()[0]
    base_name = os.path.basename(in_path)        # e.g. 0123.jpg
    out_path  = os.path.join(save_root, base_name)
    # ──────────────────────────────────────────────

    save_image(fake_B, out_path)

    if (i + 1) % 100 == 0:
        print(f'[{i+1}/{len(dataset)}] saved {base_name}')

print(f'✓ All images written to {save_root}')
