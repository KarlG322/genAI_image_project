'''
Full disclosure, I had an LLM help me make this.
That said, I did my best to make sure its understandable / that I understand it.
I also made sure it's doing things I agree with / think make sense for this project.
'''

import math
import os, glob, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

#image_size = 128 was originally a setting but has been removed.
#I think 64 is too small to see much.
#128 and 256 could both work but since this is a small project 128 is probably fine.
#More than 256 would be computationally expensive and would require lots of training data.
#Note that a lot of the comments here assume 128x128 resolution but it should work on 256x256 as well assuming everything fits in memory

image_directory = "/content/drive/MyDrive/MSAI/spring_2026_quarter/GenAI/image_generation/PNGs_128_9953"
checkpoint_directory = "/content/drive/MyDrive/MSAI/spring_2026_quarter/GenAI/image_generation/checkpoints_9953"
latest_checkpoint = os.path.join(checkpoint_directory, "latest.pt")
learning_rate = 2e-4
total_steps = 100000
diffusion_timesteps = 1000
checkpoint_interval = 5000
batch_size = 64 #64 works for 128x128

#building the dataset
#This is different from my initial version in that it immediately loads everything to the GPU and no longer uses a pytorch dataloader
#And I changed num_workers to 0 at the same time because they were causing an error that prevented training
def load_images_to_gpu(img_dir):
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png"))) #find all .png files in the directory
    print(f"Pre-loading {len(paths)} images into GPU memory...")
    to_tensor = transforms.Compose([transforms.ToTensor()])
    images = torch.stack([
        to_tensor(Image.open(p).convert("L")) for p in paths
    ])  #This creates a tensor on the CPU with shape[N, 1, 128, 128]
    images = (images - 0.5) / 0.5  #The tensor values are pre-normalized from [0, 1] to [-1, 1] since that's what the diffusion math wants
    images = images.to("cuda")
    print(f"  loaded {images.element_size() * images.nelement() / 1e6:.1f} MB to GPU")
    return images

images = load_images_to_gpu(image_directory)
N = images.shape[0]
print(f"Dataset: {N} images")

#defining the model
class ResBlock(nn.Module):
    #This is used several times in the diffusion model.
    #It takes as input x which is a feature map which includes batch, input channels, height, and width. It also takes a timestep embedding. These can be seen as the inputs in forward.
    #It does a group norm to mean 0 variance 1 (splitting into 8 groups and doing that for each of them separately)
    #It then does a ReLU activation and a convolutional layer.
    #Then h = h + self.time_mlp(F.relu(t_emb))[:, :, None, None] adds in the timestep information which is initially in a 256 length vector but is passed through a linear layer which converts it to out_ch dimensions then those are added to the feature map from the previous layer.
    #The [:, :, None, None] adds extra dimensions so this can be broadcast to every height and width location
    #Then there's another normalization, activation, and convolutional layer
    #then there's a skip connection
    #and that's it
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.relu(self.norm1(x)))
        h = h + self.time_mlp(F.relu(t_emb))[:, :, None, None]
        h = self.conv2(F.relu(self.norm2(h)))
        return h + self.skip(x)

class UNet(nn.Module):
    def __init__(self, channels=(64, 128, 256, 256), time_dim=256, n_timesteps=diffusion_timesteps):
        super().__init__()
        self.time_embed = nn.Embedding(n_timesteps, time_dim)
        c1, c2, c3, c4 = channels
        self.in_conv = nn.Conv2d(1, c1, 3, padding=1)
        self.down1   = ResBlock(c1, c1, time_dim)
        self.pool1   = nn.Conv2d(c1, c1, 4, stride=2, padding=1)
        self.down2   = ResBlock(c1, c2, time_dim)
        self.pool2   = nn.Conv2d(c2, c2, 4, stride=2, padding=1)
        self.down3   = ResBlock(c2, c3, time_dim)
        self.pool3   = nn.Conv2d(c3, c3, 4, stride=2, padding=1)
        self.mid     = ResBlock(c3, c4, time_dim)
        self.up3     = ResBlock(c4 + c3, c3, time_dim)
        self.up2     = ResBlock(c3 + c2, c2, time_dim)
        self.up1     = ResBlock(c2 + c1, c1, time_dim)
        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t) #This looks up the embedding vector for each timestep in the batch. It has shape [batch, 256] and represents the current noise level.
        h1 = self.down1(self.in_conv(x), t_emb) #convolution to 64 channels then a ResBlock on that. This takes us from [batch, 1, 128, 128] to [batch, 64, 128, 128]
        h2 = self.down2(self.pool1(h1), t_emb) #stride 2 convolution that takes us to [batch, 64, 64, 64] then ResBlock raising channels to [batch, 128, 64, 64]
        h3 = self.down3(self.pool2(h2), t_emb) #same as immediately above, this time taking us to [batch, 256, 32, 32]
        m  = self.mid(self.pool3(h3), t_emb) #same as immediately above except that the ResBlock doesn't double the channels. Output is [batch, 256, 16, 16]. This is the bottleneck.
        u3 = F.interpolate(m,  scale_factor=2, mode="nearest") #upsample to [batch, 256, 32, 32]
        u3 = self.up3(torch.cat([u3, h3], dim=1), t_emb) #torch.cat([u3, h3], dim=1) concatenates that with the output of h3 which is 1 of the earlier layers (this is a skip connection). Shape: [batch, 256+256, 32, 32] so [batch, 512, 32, 32]. Then ResBlock to [batch, 256, 32, 32].
        u2 = F.interpolate(u3, scale_factor=2, mode="nearest") #upsample to [batch, 256, 64, 64]
        u2 = self.up2(torch.cat([u2, h2], dim=1), t_emb) #concatenate with h2 to [batch, 384, 64, 64]. ResBlock to [batch, 128, 64, 64].
        u1 = F.interpolate(u2, scale_factor=2, mode="nearest") #upsample to [batch, 128, 128, 128]
        u1 = self.up1(torch.cat([u1, h1], dim=1), t_emb) #concatenate with h1 to [batch, 192, 128, 128]. ResBlock to [batch, 64, 128, 128].
        return self.out_conv(F.relu(self.out_norm(u1))) #groupnorm, a relu, then 1 last convolution to [batch, 1, 128, 128]

model = UNet().to("cuda")

#diffusion schedule

#using a cosine diffusion schedule with no offset.
#Using an offset would probably be slightly better performance but I didn't think it was necessary for this project and not including it makes this part a bit simpler
def cosine_alpha_bar(t, T):
    return torch.cos(t / T * math.pi / 2) ** 2 #this is the formula to generate alpha bar which is the amount of signal in the image at a particular timestep
t_indices = torch.arange(diffusion_timesteps + 1, device="cuda", dtype=torch.float32) #this is just a 1D tensor that counts from 0-1000.
abar = cosine_alpha_bar(t_indices, diffusion_timesteps) #This is a 1D tensor that includes, for each timestep from 0-1000, the amount of signal in the image at that timestep
betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999) #The math here is doing division like this [ab(1)/ab(0), ab(2)/ab(1), ..., ab(1000)/ab(999)] which generates betas from alpha bars. The clamp here prevents the betas (the % noise) from ever being 100% which would completely remove the signal.

#This part of the code calculates alpha
#I realize this part is a bit odd in that we calculated betas from alpha bar and now we're calculating alpha bar from betas. I'm just keeping this part because it's what my code did when I was using my original linear diffusion schedule and the performance impact should be negligible. Also the alpha bars used above and here are slightly different due to the clamp.
alphas = 1.0 - betas #alphas is how much of the previous step is preserved
alpha_bar = torch.cumprod(alphas, dim=0) #This is the cumulative product of the alphas up to a timestep. It allows us to avoid calculating all the way up to xt we can just do it all in 1 go
sqrt_ab    = torch.sqrt(alpha_bar) #This and the line below are just calculating coefficients that will be used when adding noise so more efficient to do it once here
sqrt_1mab  = torch.sqrt(1.0 - alpha_bar)

#This adds noise to clean image x0.
#Noise is epsilon.
#The sqrt terms are the coefficents precomputed above
#t is originally a 1 dimensional tensor of shape [batch_size], one per image.
#sqrt_ab[t] changes this to have shape [batch_size].
#The [:, None, None, None] then makes it [batch_size, 1, 1, 1] so it can broadcast against x0 which is [batch_size, channels, height, width].
def add_noise(x0, t, noise):
    return sqrt_ab[t][:, None, None, None] * x0 + sqrt_1mab[t][:, None, None, None] * noise

#helper functions for checkpointing
def save_checkpoint(step, model, opt, scaler):
    state = {"step": step, "model": model.state_dict(),
             "opt": opt.state_dict(), "scaler": scaler.state_dict()}
    tmp = latest_checkpoint + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, latest_checkpoint)
    torch.save(state, os.path.join(checkpoint_directory, f"step_{step:06d}.pt"))

def load_checkpoint(model, opt, scaler):
    if not os.path.exists(latest_checkpoint):
        return 0
    state = torch.load(latest_checkpoint, map_location="cuda")
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    scaler.load_state_dict(state["scaler"])
    print(f"Resumed from step {state['step']}")
    return state["step"]

#training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler("cuda", enabled=True)
step = load_checkpoint(model, optimizer, scaler)

t0 = time.time() #this is for printing progress updates
losses = []


while step < total_steps:
    #this picks a random batch of indices to train on
    idx = torch.randint(0, N, (batch_size,), device="cuda") #N is number of images and is defined when I make the dataset above
    x0 = images[idx]

    #some minimal data augmentation that just does a horizontal flip 50% of the time
    flip_mask = torch.rand(x0.size(0), device="cuda") < 0.5
    x0 = torch.where(flip_mask[:, None, None, None], x0.flip(-1), x0)

    t = torch.randint(0, diffusion_timesteps, (x0.size(0),), device="cuda")
    noise = torch.randn_like(x0)
    xt = add_noise(x0, t, noise)

    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=True):
        pred = model(xt, t)
        loss = F.mse_loss(pred, noise)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    losses.append(loss.item())
    step += 1

    #printing progress updates
    if step % 100 == 0:
        average_loss = sum(losses[-100:]) / 100
        time_per_100 = 100 / (time.time() - t0)
        t0 = time.time()
        print(f"step {step:>6}/{total_steps}  loss {average_loss:.4f}  ({time_per_100:.1f} it/s)")

    #saving checkpoints
    if step % checkpoint_interval == 0 or step == total_steps:
        save_checkpoint(step, model, optimizer, scaler)
        print("Checkpoint saved")

    if step >= total_steps:
        break

print("Done.")