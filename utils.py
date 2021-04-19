import random
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2



def gen_input_mask(shape):
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape    
    for i in range(bsize):
        rand = np.random.randint(100)
        if rand%2:
            mask[i,:,:,0:mask_w//2] = torch.ones(mask_h,mask_w//2)
        else:
            mask[i,:,:,mask_w//2:] = torch.ones(mask_h,mask_w//2)

    return mask



def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))


def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin: ymin + h, xmin: xmin + w]


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)


def rejoiner(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network, whose shape = (N, 1, H, W).
        - output (torch.Tensor, required)
                Output tensor of Completion Network, whose shape = (N, 1, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network, whose shape = (N, 1, H, W).
    * returns:
                Output image tensor of shape (N, 1, H, W) inpainted with poisson image editing method.
    """
    input = input.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    num_samples = input.shape[0]
    inp_width = input.shape[3]

    for i in range(num_samples):
        if not mask[i,0,0,0]:
            output[i,:,:,:inp_width//2] = input[i,:,:,:inp_width//2]
        else:
            output[i,:,:,inp_width//2:] = input[i,:,:,inp_width//2:]

    return output
