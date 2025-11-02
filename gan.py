
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset for flat directory
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):  
        self.folder_path = folder_path
        self.image_filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(self.image_filenames) == 0:
            raise ValueError(f"No images found in directory: {folder_path}")
        print(f"Found {len(self.image_filenames)} images in {folder_path}")
        self.transform = transform

    def __len__(self):  # תיקון len
        return len(self.image_filenames)

    def __getitem__(self, idx):  # תיקון getitem
        img_path = os.path.join(self.folder_path, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # No labels for GANs

# Paths and hyperparameters
data_dir = r"/home/student/102flowers/jpg"  # Update with your path
image_size = 64
batch_size = 64
latent_dim = 100
epochs = 150

# Transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Dataset and DataLoader
dataset = CustomDataset(folder_path=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Weight Initialization
def weights_init_normal(m):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Model
class Generator(nn.Module):
    def __init__(self, latent_dim):  # תיקון init
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):  # תיקון init
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1, 1).squeeze(1)

# Initialize models
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Loss and optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Lists to store losses
discriminator_losses = []
generator_losses = []
avg_d_losses = []
avg_g_losses = []

# Training loop
for epoch in range(epochs):
    epoch_d_losses = []
    epoch_g_losses = []

    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        batch_size = imgs.size(0)

        # Real and fake labels
        valid = torch.ones(batch_size, device=device) * 0.9  # Label smoothing
        fake = torch.zeros(batch_size, device=device) + 0.1

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(imgs), valid)

        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        gen_imgs = generator(z)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Record losses for iterations
        discriminator_losses.append(d_loss.item())
        generator_losses.append(g_loss.item())

        # Save generated samples
        if i % 50 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        num_images_to_save = 25
        z = torch.randn(num_images_to_save, latent_dim, 1, 1, device=device)
        gen_imgs = generator(z)
        normalized_gen_imgs = (gen_imgs + 1) / 2
        save_image(normalized_gen_imgs, f"HW4_MIREL-generated_images_epoch_{epoch}.png", nrow=5)

    # Save losses
    avg_d_losses.append(np.mean(epoch_d_losses))
    avg_g_losses.append(np.mean(epoch_g_losses))


from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import torch

def visualize_latent_space_with_images(generator, latent_dim, device, num_samples=100):
    """
    Visualizes the latent space with the generated images embedded in the graph.
    """
    generator.eval()

    # Step 1: Generate latent vectors and images
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, 1, 1).to(device)  # Latent vectors
        gen_imgs = generator(z).cpu()  # Generate images
        gen_imgs = (gen_imgs + 1) / 2  # Normalize images to [0, 1] for display

    # Step 2: Flatten latent vectors and apply PCA
    z_flat = z.view(num_samples, -1).cpu().numpy()  # Flatten latent vectors
    pca = PCA(n_components=2)  # Reduce to 2D
    reduced_latent_space = pca.fit_transform(z_flat)

    # Step 3: Plot latent space with images
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, (x, y) in enumerate(reduced_latent_space):
        img = gen_imgs[i].permute(1, 2, 0).numpy()  # Convert image to (H, W, C) for display
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        imagebox = OffsetImage(img, zoom=0.4)  # Create image thumbnail
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)  # Place image on graph
        ax.add_artist(ab)

    ax.set_title("Latent Space Visualization with Generated Images", fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.set_xlim(reduced_latent_space[:, 0].min() - 1, reduced_latent_space[:, 0].max() + 1)
    ax.set_ylim(reduced_latent_space[:, 1].min() - 1, reduced_latent_space[:, 1].max() + 1)
    plt.grid()
    plt.show()


visualize_latent_space_with_images(generator, latent_dim, device, num_samples=100)





import torch
import pickle

# שמירת ה-state_dict עם pickle
with open('/home/student/HW4_generator_weights_final_mirel.pkl', 'wb') as f:
    pickle.dump(generator.state_dict(), f)

with open('/home/student/HW4_discriminator_weights_final_mirel.pkl', 'wb') as f:
    pickle.dump(discriminator.state_dict(), f)

# In[262]:

def load_pickle_as_state_dict(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        return pickle.load(f)

# טעינת המשקלים למודל
generator.load_state_dict(load_pickle_as_state_dict('/home/student/HW4_generator_weights_final_mirel.pkl'))
discriminator.load_state_dict(load_pickle_as_state_dict('/home/student/HW4_discriminator_weights_final_mirel.pkl'))



def load_pickle_as_state_dict(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        return pickle.load(f)
 

epochs = range(1, len(discriminator_losses) // len(dataloader) + 1)
avg_d_losses = [np.mean(discriminator_losses[i * len(dataloader):(i + 1) * len(dataloader)]) for i in range(len(epochs))]
avg_g_losses = [np.mean(generator_losses[i * len(dataloader):(i + 1) * len(dataloader)]) for i in range(len(epochs))]

plt.figure(figsize=(10, 5))
plt.plot(epochs, avg_d_losses, label='Discriminator Loss')
plt.plot(epochs, avg_g_losses, label='Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.show()


# In[260]:


plt.figure(figsize=(10, 5))
plt.plot(discriminator_losses, label='Discriminator Loss', alpha=0.7)
plt.plot(generator_losses, label='Generator Loss', alpha=0.7)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.show()