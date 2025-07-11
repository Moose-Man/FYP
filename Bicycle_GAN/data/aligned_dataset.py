import os
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class AlignedDataset(BaseDataset):
    """Paired dataset stored in *separate* A and B directories.

    Directory layout expected:
        dataroot/
            he/ train/ *.png   (domain A)
            ihc/ train/ *.png  (domain B)
            he/  test/ *.png
            ihc/ test/ *.png
    Filenames must match 1-to-1 between the domains.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        # build absolute paths to A and B domain folders
        self.dir_A = os.path.join(opt.dataroot, 'HE_resized_nocrop',  opt.phase)
        self.dir_B = os.path.join(opt.dataroot, 'IHC_resized_nocrop', opt.phase)
        
        # list and sort files in each
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        # simple filename check
        assert len(self.A_paths) == len(self.B_paths), \
            'The two domains do not contain the same number of images'

        # build a dict { filename → full_path } for fast lookup
        fname2B = {os.path.basename(p): p for p in self.B_paths}
        self.paired_paths = []
        for a_path in self.A_paths:
            fname = os.path.basename(a_path)
            assert fname in fname2B, f'No matching file for {fname} in domain B folder'
            self.paired_paths.append((a_path, fname2B[fname]))

        # channel bookkeeping
        self.input_nc  = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
        self.output_nc = opt.input_nc if opt.direction == 'BtoA' else opt.output_nc

    def __getitem__(self, index):
        A_path, B_path = self.paired_paths[index]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply SAME random transform
        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc  == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A_img)
        B = B_transform(B_img)

        return {'A': A, 'B': B,
                'A_paths': A_path,
                'B_paths': B_path}

    def __len__(self):
        return len(self.paired_paths)
