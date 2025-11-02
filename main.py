# %%
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import pickle, json
import os

import matplotlib.pyplot as plt


from gan import device, Generator,latent_dim,Discriminator,load_pickle_as_state_dict

device = device


from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

def reproduce_hw4(generator_weights_path, discriminator_weights_path, num_images=24, save_path="/home/student/"):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the models
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Load model weights from `.pkl` files
    generator.load_state_dict(load_pickle_as_state_dict(generator_weights_path))
    discriminator.load_state_dict(load_pickle_as_state_dict(discriminator_weights_path))

    generator.eval()
    discriminator.eval()
    
    # Generate new images
   
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, 1, 1).to(device)
        gen_imgs = generator(z)

    # Save the generated images
    os.makedirs(save_path, exist_ok=True)
    save_image((gen_imgs + 1) / 2, os.path.join(save_path, "HW4_generated_images_final.png"), nrow=8, normalize=True)


    # Display the generated images

    plt.figure(figsize=(18, 18))
    for j, img in enumerate(gen_imgs):
        img = img.cpu().detach()
        img = (img + 1) / 2
        img = img.permute(1, 2, 0)
        plt.subplot(6, 4, j + 1)
        plt.imshow(img.numpy().squeeze())
        plt.axis('off')
    plt.show()






if __name__ == "__main__":
    reproduce_hw4("/home/student/HW4_generator_weights_final_mirel.pkl", 
              "/home/student/HW4_discriminator_weights_final_mirel.pkl")


    