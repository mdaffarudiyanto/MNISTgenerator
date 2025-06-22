import streamlit as st
import torch
import numpy as np
from PIL import Image

torch.classes.__path__ = []

MODEL_PATH = "mnist_cgan_G.pt"
NZ         = 100   # latent length – must match training
EMB_DIM    = 50    # ditto
N_SAMPLES  = 5     # always show five

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = torch.nn.Embedding(10, EMB_DIM)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(NZ + EMB_DIM, 256, bias=False),
            torch.nn.BatchNorm1d(256), torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 512, bias=False),
            torch.nn.BatchNorm1d(512), torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 1024, bias=False),
            torch.nn.BatchNorm1d(1024), torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(1024, 1 * 28 * 28), torch.nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, self.label_emb(labels)], dim=1)
        img = self.net(x)
        return img.view(-1, 1, 28, 28)

@st.cache_resource(show_spinner=False)
def load_generator(path=MODEL_PATH):
    G = Generator()
    G.load_state_dict(torch.load(path, map_location="cpu"))
    G.eval()
    return G

def tensor_to_pil(t):
    """Convert GAN output in [-1,1] → PIL (112×112)"""
    t = (t.clamp(-1, 1) + 1) / 2 * 255      # → [0,255]
    img = t.squeeze(0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(img, mode="L").resize((112, 112), Image.NEAREST)

st.set_page_config(page_title="MNIST Generator (GAN)", layout="centered")
st.title("MNIST Digit Generator with GAN")

digit = st.selectbox("Choose a digit", list(range(10)), format_func=str)

st.markdown(f"#### Generated images of digit **{digit}**")  

# Every rerun gets a fresh batch
G = load_generator()
z = torch.randn(N_SAMPLES, NZ)
labels = torch.full((N_SAMPLES,), digit, dtype=torch.long)
with torch.no_grad():
    fakes = G(z, labels)

# Show the five samples in one centered row
cols = st.columns(N_SAMPLES, gap="small")
for img_t, c in zip(fakes, cols):
    c.image(tensor_to_pil(img_t), use_container_width=True)

