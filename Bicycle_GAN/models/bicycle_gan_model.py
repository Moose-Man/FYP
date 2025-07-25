import torch
from .base_model import BaseModel
from . import networks
from .stn import STN
from .patchnce import PatchEncoder, PatchNCELoss            
import torch.nn.functional as F

class BiCycleGANModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # -------------------------------- PatchNCE --------------------------------
        if opt.lambda_nce > 0:
            self.netPatch = networks.init_net(
                PatchEncoder(in_ch=opt.output_nc),
                gpu_ids=self.gpu_ids,
                init_type=opt.init_type, init_gain=opt.init_gain)
            self.criterionNCE = PatchNCELoss(opt.temperature_nce,
                                            opt.num_negatives_nce)
        else:
            from .patchnce import IdentityPatchNCELoss
            self.criterionNCE = IdentityPatchNCELoss()

        if opt.isTrain:
            assert opt.batch_size % 2 == 0  # load two images at one time.

        # specify the training losses you want to print out.
        self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'G_L1', 'z_L1', 'kl']
        if opt.lambda_stn > 0:
            self.loss_names.append('stn')
        if opt.lambda_nce > 0:                
            self.loss_names.append('nce')
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A_encoded', 'real_B_encoded', 'fake_B_random', 'fake_B_encoded']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        use_E = opt.isTrain or not opt.no_encode
        use_vae = True
        self.model_names = ['G']
        if opt.lambda_nce > 0:             
            self.model_names.append('Patch')
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                           init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        else:
            self.netD2 = None
        if use_E:
            self.model_names += ['E']
            self.netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, vaeLike=use_vae)

        # --- STN: if turned on, create it and add its params to G’s optimizer ---
        if opt.lambda_stn > 0:
            self.model_names.append('STN')
            self.netSTN = networks.init_net(
                STN(), gpu_ids=self.gpu_ids,
                init_type=opt.init_type, init_gain=opt.init_gain
            )
            paramsG = list(self.netG.parameters()) + list(self.netSTN.parameters())
        else:
            paramsG = list(self.netG.parameters())

        # — if PatchNCE is enabled, include its encoder in the G optimizer —
        if opt.lambda_nce > 0:
            paramsG += list(self.netPatch.parameters())

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionZ = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(paramsG, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)

    def is_train(self):
        """check if the current batch is good for training."""
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.detach().to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def test(self, z0=None, encode=False):
        with torch.no_grad():
            if encode:  # use encoded z
                z0, _ = self.netE(self.real_B)
            if z0 is None:
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
            self.fake_B = self.netG(self.real_A, z0)
            return self.real_A, self.fake_B, self.real_B

    def forward(self):
        # get real images
        half_size = self.opt.batch_size // 2
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_A_random = self.real_A[half_size:]
        self.real_B_random = self.real_B[half_size:]
        # get encoded z
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)
        # get random z
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)
        # generate fake_B_encoded
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG(self.real_A_encoded, self.z_random)

        if self.opt.lambda_nce > 0:
            self.feat_q = self.netPatch(self.fake_B_encoded)
            with torch.no_grad():
                self.feat_k = self.netPatch(self.real_B_encoded)

        if self.opt.conditional_D:
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded.detach()], 1)
            self.fake_data_random  = torch.cat([self.real_A_encoded, self.fake_B_random ], 1)
            self.real_data_random  = torch.cat([self.real_A_random,  self.real_B_random.detach()], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random  = self.fake_B_random
            self.real_data_encoded = self.real_B_encoded.detach()
            self.real_data_random  = self.real_B_random.detach()

        # compute z_predict
        if self.opt.lambda_z > 0.0:
            self.mu2, logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate

    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.opt.lambda_GAN)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5 * self.opt.lambda_kl)
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            # (1) absolute error
            diff = torch.abs(self.fake_B_encoded - self.real_B_encoded)  # (N/2,C,H,W)

            # (2) take only the mask rows that correspond to the *encoded* half-batch
            m = self.mask[:diff.size(0)]            # (N/2,1,H,W)
            m = m.expand_as(diff)                   # broadcast from 1→C channels

            # (3) masked mean-absolute-error
            self.loss_G_L1 = (diff * m).sum() / (m.sum() + 1e-6)
            self.loss_G_L1 *= self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        # 4. PatchNCE contrastive loss
        if self.opt.lambda_nce > 0:
            self.loss_nce = self.criterionNCE(self.feat_q, self.feat_k) * self.opt.lambda_nce
        else:
            self.loss_nce = 0.0
        self.loss_G = (self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 +
                    self.loss_kl + getattr(self, 'loss_stn', 0.0) +
                    getattr(self, 'loss_nce', 0.0))

     
        
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad([self.netD, self.netD2], True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = self.criterionZ(self.mu2, self.z_random) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad([self.netD, self.netD2], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()

        # update G alone
        if self.opt.lambda_z > 0.0:
            self.set_requires_grad([self.netE], False)
            self.backward_G_alone()
            self.set_requires_grad([self.netE], True)

        self.optimizer_E.step()
        self.optimizer_G.step()

    def optimize_parameters(self):
        # 1) STN warp on the raw inputs (so both encode() and recon use the warped B)
        if self.opt.lambda_stn > 0:
            # warp entire batch of real_B, get theta & sampling grid
            warped_B, theta, grid = self.netSTN(self.real_A, self.real_B)
            # build mask = 1 inside the valid sampling region [-1,1]×[-1,1]
            mask = (grid.abs() <= 1).all(dim=-1, keepdim=True).float()    # (N,H,W,1)
            mask = mask.permute(0, 3, 1, 2)                              # (N,1,H,W)
            # override real_B with warped version and save mask
            self.real_B = warped_B
            self.mask   = mask
            # STN identity‐theta regularisation
            id_theta = (torch.tensor([1.,0.,0., 0.,1.,0.], device=self.device)
                        .view(1,2,3)
                        .expand(theta.size(0), -1, -1))
            self.loss_stn = 0.5 * self.opt.lambda_stn * F.mse_loss(theta, id_theta)
        else:
            self.mask = torch.ones_like(self.real_B[:, :1])   # full 1-mask
            self.loss_stn = 0.0

        # 2) Forward pass: encode & generate using the warped real_B internally
        self.forward()

        # 3) Update G & E (backward_EG now includes self.loss_stn in self.loss_G)
        self.update_G_and_E()

        # 4) Update discriminators
        self.update_D()


