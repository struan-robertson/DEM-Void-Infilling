import os
import torch
import torch.nn as nn
from torch import autograd

from networks import Generator, LocalDis, GlobalDis
from tools import get_model_list, local_patch, spatial_discounting_mask


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config["cuda"]

        self.netG = Generator(self.config, self.use_cuda)
        self.localD = LocalDis(self.config, self.use_cuda)
        self.globalD = GlobalDis(self.config, self.use_cuda)

        self.optimizer_g = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.config["lr"],
            betas=(self.config["beta1"], self.config["beta2"]),
        )
        d_params = list(self.localD.parameters()) + list(self.globalD.parameters())
        self.optimizer_d = torch.optim.Adam(
            d_params,
            lr=config["lr"],
            betas=(self.config["beta1"], self.config["beta2"]),
        )
        if self.use_cuda:
            self.netG = self.netG.cuda()
            self.localD = self.localD.cuda()
            self.globalD = self.globalD.cuda()

    def forward(self, x, bboxes, masks, ground_truth, compute_loss_g=False):
        self.train()
        l1_loss = nn.L1Loss()
        losses = {}

        x1, x2 = self.netG(x, masks)
        local_patch_gt = local_patch(ground_truth, bboxes)
        x1_inpaint = x1 * masks + x * (1.0 - masks)
        x2_inpaint = x2 * masks + x * (1.0 - masks)
        local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)
        local_patch_x2_inpaint = local_patch(x2_inpaint, bboxes)

        # D part
        # wgan d loss
        local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
            self.localD, local_patch_gt, local_patch_x2_inpaint.detach()
        )
        global_real_pred, global_fake_pred = self.dis_forward(
            self.globalD, ground_truth, x2_inpaint.detach()
        )
        losses["wgan_d"] = (
            torch.mean(local_patch_fake_pred - local_patch_real_pred)
            + torch.mean(global_fake_pred - global_real_pred)
            * self.config["global_wgan_loss_alpha"]
        )
        # gradients penalty loss
        local_penalty = self.calc_gradient_penalty(
            self.localD, local_patch_gt, local_patch_x2_inpaint.detach()
        )
        global_penalty = self.calc_gradient_penalty(
            self.globalD, ground_truth, x2_inpaint.detach()
        )
        losses["wgan_gp"] = local_penalty + global_penalty

        # G part
        if compute_loss_g:
            sd_mask = spatial_discounting_mask(self.config)
            losses["l1"] = l1_loss(
                local_patch_x1_inpaint * sd_mask, local_patch_gt * sd_mask
            ) * self.config["coarse_l1_alpha"] + l1_loss(
                local_patch_x2_inpaint * sd_mask, local_patch_gt * sd_mask
            )
            losses["ae"] = l1_loss(
                x1 * (1.0 - masks), ground_truth * (1.0 - masks)
            ) * self.config["coarse_l1_alpha"] + l1_loss(
                x2 * (1.0 - masks), ground_truth * (1.0 - masks)
            )

            # wgan g loss
            local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
                self.localD, local_patch_gt, local_patch_x2_inpaint
            )
            global_real_pred, global_fake_pred = self.dis_forward(
                self.globalD, ground_truth, x2_inpaint
            )
            losses["wgan_g"] = (
                -torch.mean(local_patch_fake_pred)
                - torch.mean(global_fake_pred) * self.config["global_wgan_loss_alpha"]
            )

        return losses, x2_inpaint

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

        return real_pred, fake_pred

    # Calculate gradient penalty
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def inference(self, x, masks):
        self.eval()
        x1, x2, offset_flow = self.netG(x, masks)
        # x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1.0 - masks)

        return x2_inpaint, offset_flow

    def save_model(self, checkpoint_dir, iteration):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, "gen_%08d.pt" % iteration)
        dis_name = os.path.join(checkpoint_dir, "dis_%08d.pt" % iteration)
        opt_name = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(self.netG.state_dict(), gen_name)
        torch.save(
            {"localD": self.localD.state_dict(), "globalD": self.globalD.state_dict()},
            dis_name,
        )
        torch.save(
            {
                "gen": self.optimizer_g.state_dict(),
                "dis": self.optimizer_d.state_dict(),
            },
            opt_name,
        )

    def resume(self, checkpoint_dir, epoch=1):
        checkpoint_dir = os.path.join(checkpoint_dir, "saved_models")

        # Get file names from epoch
        epoch_string = "00000000"
        epoch_string = epoch_string[: 8 - len(str(epoch))] + str(epoch) + ".pt"

        gen_string = os.path.join(checkpoint_dir, "gen_" + epoch_string)
        dis_string = os.path.join(checkpoint_dir, "dis_" + epoch_string)
        opt_string = os.path.join(checkpoint_dir, "optimizer.pt")

        # Generator
        self.netG.load_state_dict(torch.load(gen_string))

        # Discriminator
        state_dict = torch.load(dis_string)
        self.localD.load_state_dict(state_dict["localD"])
        self.globalD.load_state_dict(state_dict["globalD"])

        # Optimizers
        state_dict = torch.load(opt_string)
        self.optimizer_d.load_state_dict(state_dict["dis"])
        self.optimizer_g.load_state_dict(state_dict["gen"])

        print("Resume from {} at epoch {}".format(checkpoint_dir, epoch))

        return epoch
