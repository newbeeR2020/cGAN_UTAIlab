import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ハイパーパラメータ
latent_dim = 10
n_classes = 10
img_size = 28

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator の定義
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# モデルのロード
generator = Generator(latent_dim, n_classes).to(device)
generator.load_state_dict(torch.load("GANgenerator.pth", map_location=device))
generator.eval()

st.title("cGAN Image Generator")
st.write("Enter a number (0-9) to generate an image of that digit.")

label = st.number_input("Enter Label (0-9)", min_value=0, max_value=9, value=0, step=1)

if st.button("Generate Image"):
    z = torch.randn(1, latent_dim, device=device)
    gen_labels = torch.tensor([label], dtype=torch.long, device=device)
    
    with torch.no_grad():
        gen_img = generator(z, gen_labels)
    
    gen_img = (gen_img + 1) / 2  # [-1,1] -> [0,1]
    gen_img = gen_img.squeeze().cpu().numpy()
    
    fig, ax = plt.subplots()
    ax.imshow(gen_img, cmap='gray')
    ax.set_title(f"Generated Image for Label {label}")
    ax.axis('off')
    
    st.pyplot(fig)
