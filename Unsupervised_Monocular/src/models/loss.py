import torch
from torch import nn
import torch.nn.functional as F


class MonoDepthLoss(nn.Module):
    def __init__(self, c_ap_alpha=0.85, smoothness_weight=1.0, lr_consistency_weight=1.0):
        """
        MonoDepth loss function
        :param c_ap_alpha: hyper parameter alpha for ``SSIM`` loss
        """""
        super(MonoDepthLoss, self).__init__()
        self.c_ap_alpha = c_ap_alpha
        self.smoothness_weight = smoothness_weight
        self.lr_consistency_weight = lr_consistency_weight

    def scale_pyramid(self, img, num_scales):
        """
        Get scaled images
        :param img: input image [b, 3, h, w]
        :param num_scales: number of level to scale
        :return: list of scaled images [b, 3, nh, nw]
        """""
        scaled_imgs = [img]
        h = img.size(2)
        w = img.size(3)
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img, size=[nh, nw], mode='bilinear', align_corners=True))
        return scaled_imgs

    def get_reconstruction_loss(self):
        """
        Calculate reconstruction loss ``C_ap`` from referenced paper
        :return: None
        """""
        # L1
        l1_left = [torch.mean(torch.abs(self.left_est[i] - self.left_pyramids[i])) for i in range(4)]
        l1_right = [torch.mean(torch.abs(self.right_est[i] - self.right_pyramids[i])) for i in range(4)]
        # SSIM
        ssim_left = [torch.mean(self.SSIM(self.left_est[i], self.left_pyramids[i])) for i in range(4)]
        ssim_right = [torch.mean(self.SSIM(self.right_est[i], self.right_pyramids[i])) for i in range(4)]
        # Reconstruction loss for each opposite view
        image_loss_left = [self.c_ap_alpha * ssim_left[i] + (1 - self.c_ap_alpha) * l1_left[i] for i in range(4)]
        image_loss_right = [self.c_ap_alpha * ssim_right[i] + (1 - self.c_ap_alpha) * l1_right[i] for i in range(4)]
        image_loss = sum(image_loss_left + image_loss_right)
        return image_loss

    def transform_with_disparity(self, img, disp):
        """
        Apply disparity transform to image to generate counterpart's view
        :param img: individual left image [b, 3, h, w]
        :param disp: corresponding disparity map [b, 1, h, w]
        :return: 
        """""
        batch_size, _, height, width = img.size()
        # Create mesh grid
        x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)
        # Shirt in x direction
        x_shifts = disp[:, 0, :, :]
        # Prepare for interpolation and normalize to range [-1, 1]
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)  # original range [0, 1] from linspace
        flow_field = flow_field * 2 - 1

        output = F.grid_sample(img, flow_field, mode='bilinear', align_corners=True, padding_mode='zeros')
        return output

    def generate_image_left(self, img, disp):
        """
        Generate left image from right ``img`` using ``disp``
        :param img: individual left image [b, 3, h, w]
        :param disp: corresponding disparity map [b, 1, h, w]
        :return: left image [b, 3, h, w]
        """""
        return self.transform_with_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        """
        Generate right image from left ``img`` using disp``
        :param img: individual right image [b, 3, h, w]
        :param disp: corresponding disparity map [b, 1, h, w]
        :return: right image [b, 3, h, w]
        """""
        return self.transform_with_disparity(img, disp)

    def SSIM(self, x, y):
        """
        Compute SSIM loss
        :param x: true image [b, 3, h, w]
        :param y: reconstructed image [b, 3, h, w]
        :return: ``float``
        """""
        # Constants used in SSIM loss
        # More information here: https://en.wikipedia.org/wiki/Structural_similarity_index_measure
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        ssim_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)             # numerator
        ssim_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)    # denominator
        ssim = ssim_n / ssim_d

        return torch.clamp((1 - ssim) / 2, 0, 1)  # (1 - SSIM) / 2 to prepare for C_ap

    def get_smoothness_loss(self):
        """
        Compute disparity smoothness loss ``C_ds`` from referenced paper
        :return: ``float``
        """""
        disp_left_smoothness = self.get_disparity_smoothness(self.left_est, self.left_pyramids)
        disp_right_smoothness = self.get_disparity_smoothness(self.right_est, self.right_pyramids)
        # Give less weights to lower scale by dividing by 2^scale_level
        disp_left_loss = [torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** i for i in range(4)]
        disp_right_loss = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i for i in range(4)]
        return sum(disp_left_loss + disp_right_loss)

    def get_disparity_smoothness(self, disp, pyramid):
        """
        Get disparity smoothness, equation (3) from referenced paper
        :param disp: disparity map [b, 1, h, w] x 4 scale level
        :param pyramid: scaled images [batch, 3, h, w] x 4 scale level
        :return: 
        """""
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def gradient_x(self, img):
        """
        Calculate gradient in x direction
        :param img: image to calculate gradient [batch, channel, h, w]
        :return: gradient in x direction [batch, channel, h, w]
        """""
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx

    def gradient_y(self, img):
        """
        Calculate gradient in y direction
        :param img: image to calculate gradient [batch, channel, h, w]
        :return: gradient in x direction [batch, channel, h, w]
        """""
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    def get_lr_consistency_loss(self):
        """
        Compute disparity smoothness loss ``C_lr`` from referenced paper
        :return: ``float``
        """""
        right_to_left_disp = [self.generate_image_left(self.disp_right_est[i],
                                                       self.disp_left_est[i]) for i in range(4)]
        left_to_right_disp = [self.generate_image_right(self.disp_left_est[i],
                                                        self.disp_right_est[i]) for i in range(4)]

        lr_left_loss = [torch.mean(torch.abs(right_to_left_disp[i]
                                             - left_to_right_disp[i])) for i in range(4)]
        lr_right_loss = [torch.mean(torch.abs(left_to_right_disp[i]
                                              - right_to_left_disp[i])) for i in range(4)]
        lr_loss = sum(lr_left_loss + lr_right_loss)
        return lr_loss

    def _build_outputs(self, disparities, targets):
        """
        Prepare data for loss calculation
        :param disparities: disparity map [b, 1, h, w] x 4 scale level
        :param targets: [left_images, right_images] [b, 3, h, w]
        :return: ``None``
        """""
        left_images, right_images = targets
        left_pyramids = self.scale_pyramid(left_images, 4)
        right_pyramids = self.scale_pyramid(right_images, 4)
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in disparities]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in disparities]

        self.disp_est = disparities                 # [batch, 2, h, w] x 4 scale level
        self.left_images = left_images              # [batch, 3, h, w]
        self.right_images = right_images            # [batch, 3, h, w]
        self.disp_left_est = disp_left_est          # [batch, h, w]
        self.disp_right_est = disp_right_est        # [batch, h, w]
        self.left_pyramids = left_pyramids          # [batch, 3, h, w] x 4 scale level
        self.right_pyramids = right_pyramids        # [batch, 3, h, w] x 4 scale level
        self.left_est = [self.generate_image_left(self.right_pyramids[i], self.disp_left_est[i]) for i in range(4)]
        # [batch, 3, h, w] x 4 scale level
        self.right_est = [self.generate_image_right(self.left_pyramids[i], self.disp_right_est[i]) for i in range(4)]
        # [batch, 3, h, w] x 4 scale level

    def forward(self, disparities, targets):
        """
        Feed forward function to compute loss for MonoDepth
        :param disparities: disparity outputs from 4 scales [b, 1, h, w]
        :param targets: [left_images, right_images] [b, 3, h, w]
        :return: ``float``
        """""
        self._build_outputs(disparities, targets)

        reconstruction_loss = self.get_reconstruction_loss()
        smoothness_loss = self.get_smoothness_loss()
        lr_consistency_loss = self.get_lr_consistency_loss()

        total_loss = (reconstruction_loss + smoothness_loss * self.smoothness_weight +
                      lr_consistency_loss * self.lr_consistency_weight)
        return total_loss


if __name__ == '__main__':
    loss = MonoDepthLoss()
    batch_size = 8
    disp1 = torch.rand(batch_size, 2, 256, 512)
    disp2 = torch.rand(batch_size, 2, 128, 256)
    disp3 = torch.rand(batch_size, 2, 64, 128)
    disp4 = torch.rand(batch_size, 2, 32, 64)
    disp = [disp1, disp2, disp3, disp4]
    left = torch.rand(batch_size, 3, 256, 512)
    right = torch.rand(batch_size, 3, 256, 512)
    input = [left, right]

    output = loss(disp, input)
    print(output)
