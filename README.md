# Learning Probability Density Function using GAN

## Project Overview
This project implements a Generative Adversarial Network (GAN) to learn the probability density function (PDF) of a transformed NO₂ concentration variable. The GAN learns the distribution directly from 263,627 data samples without requiring knowledge of the analytical form.

---

## Dataset
- **Source**: India Air Quality Dataset (Kaggle)
- **Feature Used**: NO₂ concentration
- **Data Processing**: Missing values removed using `.dropna()`
- **Total Samples**: 263,627 valid NO₂ measurements
- **Data Type**: Converted to `float32` for PyTorch compatibility

---

## Step 1: Data Transformation
Each NO₂ value `x` is transformed into `z` using the formula:

```
z = x + aᵣ × sin(bᵣ × x)
```

**Where:**
- **aᵣ** = 0.5 × (r mod 7) = **3.0** (for r = 102317200)
- **bᵣ** = 0.3 × ((r mod 5) + 1) = **0.3** (for r = 102317200)
- **r** = University roll number (102317200)

**Normalization**: The transformed data `z` is normalized using:
```python
z_norm = (z - z_mean) / z_std
```
This ensures stable training by keeping values in a standard range.

---

##  Step 2: GAN Architecture

### Generator Network
```
Input (1D noise) → Linear(1, 32) → ReLU
                 → Linear(32, 32) → ReLU
                 → Linear(32, 1) → Output
```
- **Input**: 1D Gaussian noise N(0,1)
- **Hidden Layers**: Two layers with 32 neurons each
- **Activation**: ReLU (Rectified Linear Unit)
- **Output**: Single continuous value representing a sample from learned distribution

```python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
```

### Discriminator Network
```
Input (1D sample) → Linear(1, 32) → LeakyReLU(0.2)
                  → Linear(32, 32) → LeakyReLU(0.2)
                  → Linear(32, 1) → Sigmoid → Output
```
- **Input**: 1D sample (real or generated)
- **Hidden Layers**: Two layers with 32 neurons each
- **Activation**: LeakyReLU with slope 0.2
- **Output Layer**: Sigmoid activation for probability output [0,1]

```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

### Training Configuration

| **Parameter** | **Value** |
|---------------|-----------|
| Epochs | 4,000 |
| Batch Size | 128 |
| Loss Function | Binary Cross Entropy (BCE) |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Device | CPU |

### Training Process
1. **Discriminator Training**:
   - Sample 128 real data points randomly
   - Generate 128 fake samples from noise
   - Train discriminator to classify real vs fake
   - Loss = BCE(D(real), 1) + BCE(D(fake), 0)

2. **Generator Training**:
   - Generate 128 new fake samples
   - Train generator to fool discriminator
   - Loss = BCE(D(fake), 1)

3. **Monitoring**: Loss values printed every 500 epochs

```python
epochs = 4000
batch_size = 128

for epoch in range(epochs):
    # Train Discriminator
    idx = torch.randint(0, z_tensor.size(0), (batch_size,))
    real = z_tensor[idx]
    
    noise = torch.randn(batch_size, 1).to(device)
    fake = G(noise).detach()
    
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    
    d_loss = criterion(D(real), real_labels) + criterion(D(fake), fake_labels)
    
    d_opt.zero_grad()
    d_loss.backward()
    d_opt.step()
    
    # Train Generator
    noise = torch.randn(batch_size, 1).to(device)
    fake = G(noise)
    
    g_loss = criterion(D(fake), real_labels)
    
    g_opt.zero_grad()
    g_loss.backward()
    g_opt.step()
```

---

## 📈 Step 3: PDF Approximation

### Sample Generation
After training, 10,000 samples are generated from the trained generator:
```python
G.eval()
with torch.no_grad():
    noise = torch.randn(10000, 1).to(device)
    gen_z = G(noise).cpu().numpy()
    gen_z = gen_z * z_std + z_mean  # Denormalize
```

### PDF Estimation Methods

**1. Histogram Comparison**
- 80 bins used for both real and generated data
- Density normalization applied (`density=True`)
- Overlapping histograms show close distribution match

**2. Kernel Density Estimation (KDE)**
- **Kernel**: Gaussian
- **Bandwidth**: 0.3
- **Sample Points**: 1,000 points across data range
- Produces smooth continuous PDF curve

```python
kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(gen_z)
z_range = np.linspace(gen_z.min(), gen_z.max(), 1000).reshape(-1, 1)
log_dens = kde.score_samples(z_range)
```

---

##  Results

### Histogram Comparison
The overlapping histograms demonstrate that the GAN successfully learned the target distribution:

<img width="680" height="451" alt="Histogram Comparison" src="https://github.com/user-attachments/assets/1e1ed117-0c22-4421-8354-ab9ce4c5c9a6" />

**Key Observations:**
- Real data (z): Shown with transparency (alpha=0.4)
- GAN samples: Shown with transparency (alpha=0.6)
- Distribution shapes closely match

### KDE Curve
The smooth KDE curve confirms accurate learning of the PDF:

<img width="707" height="470" alt="KDE PDF Estimation" src="https://github.com/user-attachments/assets/db141b54-2c34-441a-8c21-7cd79afdd1b6" />

**Key Features:**
- Captures all major distribution modes
- Shows continuous probability density
- Validates GAN's capability to approximate unknown PDFs

---

##  Performance Analysis

| **Metric** | **Observation** |
|-----------|----------------|
| Training Epochs | 4,000 iterations completed |
| Sample Size | 10,000 generated samples |
| Distribution Matching | High visual similarity in histograms |
| Mode Coverage | All major peaks successfully captured |
| Training Stability | Losses converged without divergence |
| Smoothness | KDE curve is smooth and continuous |

**Sample Training Output:**
```
Epoch 0 | D Loss: 1.3862 | G Loss: 0.6917
Epoch 500 | D Loss: 1.3833 | G Loss: 0.6974
Epoch 1000 | D Loss: 1.3288 | G Loss: 0.7293
Epoch 1500 | D Loss: 1.3161 | G Loss: 0.7450
Epoch 2000 | D Loss: 1.3201 | G Loss: 0.7463
Epoch 2500 | D Loss: 1.3842 | G Loss: 0.7015
Epoch 3000 | D Loss: 1.3895 | G Loss: 0.6966
Epoch 3500 | D Loss: 1.3930 | G Loss: 0.6916
```

---

## Key Observations

1. **Successful Learning**: The GAN successfully learned the complex transformed NO₂ distribution without any prior knowledge of its analytical form

2. **Stable Training**: Both generator and discriminator losses stabilized, indicating successful adversarial equilibrium

3. **Data Normalization**: Normalizing the transformed data (z) before training was crucial for stable GAN convergence

4. **Architecture Simplicity**: Despite simple architecture (2 hidden layers, 32 neurons each), the GAN captured the distribution effectively

5. **Practical Application**: Demonstrates GAN's utility for learning probability distributions when analytical PDFs are unknown

---
### Data Flow
```
CSV File → Pandas DataFrame → Extract NO₂ column
        → Remove NaN values → Apply transformation formula
        → Normalize data → Convert to PyTorch tensor
        → Train GAN → Generate samples → Estimate PDF
```

---

## Conclusion

This project successfully demonstrates that **Generative Adversarial Networks can effectively learn complex probability density functions purely from sample data**, without requiring knowledge of the underlying analytical distribution. 

The transformed NO₂ concentration data served as an excellent test case, showing that:

-  GANs can capture multimodal distributions
-  Simple neural network architectures can be sufficient for 1D distributions
- Normalization is critical for training stability
-  The learned distribution closely matches the empirical data

This approach has broad applications in scenarios where the PDF is unknown or too complex to model analytically, making GANs a powerful tool for distribution learning and generative modeling.

---
