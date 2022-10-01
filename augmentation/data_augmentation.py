# =============================================================================
# Import required libraries
# =============================================================================
import os
import json
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn, optim

import matplotlib.pyplot as plt
import timeit

from augmentation.gen_critic import Generator, Critic

torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
# Check if CUDA is available
# =============================================================================
train_on_GPU = torch.cuda.is_available()
if not train_on_GPU:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')
    print(torch.cuda.get_device_properties('cuda'))


# =============================================================================
# Corel-5k settings
# =============================================================================
def Corel_5k(root, annotation_path):
    with open(annotation_path) as fp:
        json_data = json.load(fp)
    samples = json_data['samples']
    #
    imgs = []
    for sample in samples:
        imgs.append(sample['image_name'])
    return imgs


imgs = Corel_5k('./Corel-5k/images', './Corel-5k/train.json')


def corel_5k_transforms(img):
    root = './Corel-5k/images'
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img_path = os.path.join(root, img)
    image = Image.open(img_path).convert("RGB")

    if image.height == 192:
        image = image.rotate(90, expand=True)

    image = transforms.Compose([
        transforms.CenterCrop((122, 186)),
        transforms.Resize((128, 192)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        ),
    ])(image)
    return image.unsqueeze(0)


def show_tensor_image(image_tensor):
    image_tensor = (image_tensor / 2) + 0.5
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:1])
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


# =============================================================================
# WGAN settings
# =============================================================================
noise_dim = 100


def get_noise(n_samples, noise_dim):
    if train_on_GPU:
        return torch.randn(n_samples, noise_dim).cuda()
    else:
        return torch.randn(n_samples, noise_dim)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def make_wgan():
    gen = Generator(noise_dim)
    crit = Critic()

    if train_on_GPU:
        gen.cuda()
        crit.cuda()

    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)

    gen_opt = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    crit_opt = optim.Adam(crit.parameters(), lr=0.0002, betas=(0.5, 0.999))

    return gen, crit, gen_opt, crit_opt


# =============================================================================
#
# =============================================================================
iteration = 1001

c_lambda = 10
crit_repeats = 5


def get_gradient(crit, real, fake, epsilon):
    mixed_images = epsilon * real + (1 - epsilon) * fake

    mixed_scores = crit(mixed_images)

    # take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    gradient = torch.flatten(gradient, 1)
    # calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    # penalize the mean squared distance of the gradient norms from 1
    penalty = (gradient_norm - 1) ** 2
    return penalty


# =============================================================================
# Save images
# =============================================================================
def save_image(fake_image, image_name, image_type):
    if image_type == 2:
        fake_image = transforms.RandomHorizontalFlip(1)(fake_image)

    img = (fake_image / 2) + 0.5
    img = img.detach().cpu()
    img = img.squeeze(0).permute(1, 2, 0)
    img = np.array(img)

    if image_type == 3:
        img = np.rot90(img, -1)

    if image_type == 1:
        print('Image 1')
        show_tensor_image(fake_image)
        plt.imsave(os.path.join('./Corel-5k/aug-images/' +
                                image_name+"_aug.jpeg"), img)
    if image_type == 2:
        print('Image 2')
        show_tensor_image(fake_image)
        plt.imsave(os.path.join('./Corel-5k/aug-images/' +
                                image_name+"_aug_hor_flip.jpeg"), img)
    if image_type == 3:
        print('Image 3')
        plt.imshow(img)
        plt.show()
        plt.imsave(os.path.join('./Corel-5k/aug-images/' +
                                image_name+"_aug_rot_90.jpeg"), img)


# =============================================================================
# Training
# =============================================================================
def train(it, img, gen, crit, gen_opt, crit_opt, image_name):
    gen.train()
    crit.train()

    generator_loss = 0
    critic_loss = 0

    # making the generating process more challenging by flipping the image
    # vertically at every iteration by very low probability
    img = transforms.RandomVerticalFlip(0.1)(img)

    if train_on_GPU:
        real_image = img.cuda()

    for _ in range(crit_repeats):
        # Update critic #
        crit_opt.zero_grad()

        fake_noise = get_noise(1, noise_dim)
        fake_image = gen(fake_noise)
        #
        crit_fake_pred = crit(fake_image.detach())
        crit_real_pred = crit(real_image)
        #
        epsilon = torch.rand(1, 1, 1, 1, requires_grad=True).cuda()
        gradient = get_gradient(
            crit, real_image, fake_image.detach(), epsilon)
        gp = gradient_penalty(gradient)
        #
        crit_loss = crit_fake_pred - crit_real_pred + c_lambda * gp
        # Update gradients
        crit_loss.backward(retain_graph=True)
        # Update optimizer
        crit_opt.step()

        critic_loss += crit_loss.item() / crit_repeats

    # Update generator #
    gen_opt.zero_grad()

    fake_noise = get_noise(1, noise_dim)
    fake_2 = gen(fake_noise)

    crit_fake_pred = crit(fake_2)
    #
    gen_loss = -1 * crit_fake_pred
    #
    gen_loss.backward()
    # Update the weights
    gen_opt.step()

    generator_loss += gen_loss.item()

    if it == 998:
        save_image(fake_image, image_name, 1)
    if it == 999:
        save_image(fake_image, image_name, 2)
    if it == 1000:
        save_image(fake_image, image_name, 3)


for i in range(0, len(imgs)):
    start = timeit.default_timer()
    #
    gen, crit, gen_opt, crit_opt = make_wgan()
    print('number: ' + str(i+1))
    print('image: ' + imgs[i])
    img = corel_5k_transforms(imgs[i])
    image_name = imgs[i].split('.')[0]
    for it in range(iteration):
        train(it, img, gen, crit, gen_opt, crit_opt, image_name)
    #
    stop = timeit.default_timer()
    print('time: {:.3f}'.format(stop - start))
