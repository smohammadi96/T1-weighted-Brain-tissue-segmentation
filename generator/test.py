from cgi import print_form
import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl
from math import log10, sqrt
import cv2

import data
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# py.arg('--experiment_dir')
py.arg('--batch_size', type=int, default=32)
test_args = py.args()
# args = py.args_from_yaml(py.join('output\\horse2zebra', 'settings.yml'))
# args.__dict__.update(test_args.__dict__)

dataset= "horse2zebra"
datasets_dir= "datasets"
load_size= 286
crop_size= 256
batch_size= 1
epochs= 20
epoch_decay= 100
lr= 0.0002
beta_1= 0.5
adversarial_loss_mode = "lsgan"
gradient_penalty_mode = "none"
gradient_penalty_weight = 10.0
cycle_loss_weight = 10.0
identity_loss_weight = 0.0
pool_size = 50

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = py.glob(py.join('T2_final_dataset/test_with_inu'), '*')
print(A_img_paths_test)
B_img_paths_test = py.glob(py.join('T2_final_dataset/test_without_inu'), '*')
A_dataset_test = data.make_dataset(A_img_paths_test, batch_size, load_size, crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
B_dataset_test = data.make_dataset(B_img_paths_test, batch_size, load_size, crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)

# model
G_A2B = module.ResnetGenerator(input_shape=(crop_size, crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(crop_size, crop_size, 3))

# resotre
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), 'output/correct_inu_last_version_combine_T2_masks_v2/checkpoints_new/').restore()


@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A


@tf.function
def sample_B2A(B):
    B2A = G_B2A(B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return B2A, B2A2B

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# run
save_dir = py.join('T2_LR_new_v2', 'A2B')
py.mkdir(save_dir)
save_dir = py.join('T2_LR_new_v2', 'three_plot')
py.mkdir(save_dir)
i = 0
ps = []
for A in A_dataset_test:
    A2B, A2B2A = sample_A2B(A)
    for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
        img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B2A_i.numpy()], axis=1)

        volume = A2B_i.numpy()
        v_max = np.max(volume)
        v_min = np.min(volume)
        volume_norm = (volume - v_min) / (v_max - v_min)
        volume_norm2 = (volume_norm * 255).astype("int")

        volume = A_i.numpy()
        v_max = np.max(volume)
        v_min = np.min(volume)
        volume_norm = (volume - v_min) / (v_max - v_min)
        volume_norm3 = (volume_norm * 255).astype("int")
        # mm = A_i.split('\\')[-1]
        # print(np.unique(img))
        volume = img
        v_max = np.max(volume)
        v_min = np.min(volume)
        volume_norm = (volume - v_min) / (v_max - v_min)
        volume_norm = (volume_norm * 255).astype("int")
        psnr_value = PSNR(volume_norm3, volume_norm2)
        print(psnr_value)
        ps.append(psnr_value)
        cv2.imwrite(f'T2_LR_new_v2/three_plot/{i}.png', volume_norm)
        cv2.imwrite(f'T2_LR_new_v2/A2B/{i}.png', volume_norm2)

        # im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i])))
        i += 1
print(np.mean(np.array(ps)))
