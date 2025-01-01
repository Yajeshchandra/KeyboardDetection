import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        def block(in_feat, out_feat):
            return nn.Sequential(
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),
                nn.LeakyReLU(0.2)
            )
        
        self.model = nn.Sequential(
            block(latent_dim * 2, 256),
            block(256, 512),
            block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat([z, label_embedding], 1)
        img = self.model(x)
        return img.view(img.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.label_embedding = nn.Embedding(num_classes, int(np.prod(img_shape)))
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) * 2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_embedding = self.label_embedding(labels)
        x = torch.cat([img_flat, label_embedding], 1)
        return self.model(x)

class COCODataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file, 'r') as f:
            self.coco = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        
        self.img_to_anns = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.coco['images'])

    def __getitem__(self, idx):
        img_info = self.coco['images'][idx]
        img_path = f"{self.img_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        annotations = self.img_to_anns.get(img_info['id'], [])
        label = annotations[0]['category_id'] if annotations else 0
        
        return image, label

def train_cgan(json_file, img_dir, num_classes, img_shape=(3, 64, 64), epochs=200, batch_size=64, latent_dim=100):
    transform = transforms.Compose([
        transforms.Resize(img_shape[1:]),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    
    dataset = COCODataset(json_file, img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    generator = Generator(latent_dim, num_classes, img_shape)
    discriminator = Discriminator(num_classes, img_shape)
    
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
    
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for i, (real_imgs, labels) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            
            valid = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)
            
            if torch.cuda.is_available():
                real_imgs = real_imgs.cuda()
                labels = labels.cuda()
                valid = valid.cuda()
                fake = fake.cuda()
            
            # Train Generator
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, latent_dim).cuda()
            gen_labels = torch.randint(0, num_classes, (batch_size,)).cuda()
            gen_imgs = generator(z, gen_labels)
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = criterion(validity, valid)
            g_loss.backward()
            g_optimizer.step()
            
            # Train Discriminator
            d_optimizer.zero_grad()
            real_validity = discriminator(real_imgs, labels)
            fake_validity = discriminator(gen_imgs.detach(), gen_labels)
            real_loss = criterion(real_validity, valid)
            fake_loss = criterion(fake_validity, fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                      f"D loss: {d_loss.item():.4f} G loss: {g_loss.item():.4f}")
    
    return generator


if __name__ == "__main__":
    json_file = "annotated_dataset/train/_annotations.coco.json"
    img_dir = "annotated_dataset/train"
    num_classes = 76
    
    img_shape = (3, 64, 64)  # RGB, 64x64 pixels
    
    generator = train_cgan(json_file, img_dir, num_classes, img_shape)
    
    torch.save(generator.state_dict(), "generator.pth")
    print("Model saved to generator.pth")
    
    # Load model
    
    generator = Generator(latent_dim=100, num_classes=76, img_shape=(3, 64, 64))
    generator.load_state_dict(torch.load("generator.pth"))
    generator.eval()
    
    z = torch.randn(1, 100)
    label = torch.tensor([0])
    img = generator(z, label)
    img = img.squeeze().detach().numpy().transpose(1, 2, 0)
    img = (img + 1) / 2
    plt.imshow(img)
    plt.show()
    