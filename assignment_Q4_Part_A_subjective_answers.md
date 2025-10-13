```python
pip install torch torchvision
```

    Requirement already satisfied: torch in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (2.8.0)
    Requirement already satisfied: torchvision in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (0.23.0)
    Requirement already satisfied: filelock in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (from torch) (3.19.1)
    Requirement already satisfied: typing-extensions>=4.10.0 in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (from torch) (4.14.1)
    Requirement already satisfied: sympy>=1.13.3 in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (from torch) (1.14.0)
    Requirement already satisfied: networkx in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (from torch) (3.5)
    Requirement already satisfied: jinja2 in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (from torch) (3.1.6)
    Requirement already satisfied: fsspec in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (from torch) (2025.9.0)
    Requirement already satisfied: setuptools in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (from torch) (80.9.0)
    Requirement already satisfied: numpy in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (from torchvision) (2.3.1)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (from torchvision) (11.3.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (from sympy>=1.13.3->torch) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\ginis\appdata\local\programs\python\python312\lib\site-packages (from jinja2->torch) (3.0.2)
    Note: you may need to restart the kernel to use updated packages.
    

    
    [notice] A new release of pip is available: 24.0 -> 25.2
    [notice] To update, run: python.exe -m pip install --upgrade pip
    


```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Remove all the warnings
import warnings
warnings.filterwarnings('ignore')

import os
device =    "cpu"

# Retina display
%config InlineBackend.figure_format = 'retina'

try:
    from einops import rearrange
except ImportError:
    %pip install einops
    from einops import rearrange
```


```python
from sklearn import preprocessing
def normalize_image(image_tensor):
    image_tensor = image_tensor.float() / 255.0
    return image_tensor
# Load and preprocess image
color_img = torchvision.io.read_image("./iitgn.jpg")
color_img = color_img.to(dtype=torch.float32)  # Ensure tensor is of type float

# Normalize the image
color_img = normalize_image(color_img)
print("Min pixel value:", color_img.min().item())
print("Max pixel value:", color_img.max().item())
plt.imshow(color_img.permute(1, 2, 0).cpu().numpy())
print(color_img.shape)
```

    Min pixel value: 0.0
    Max pixel value: 1.0
    torch.Size([3, 640, 1280])
    


    
![png](Task4_PART_A_files/Task4_PART_A_2_1.png)
    



```python
crop = torchvision.transforms.CenterCrop(300)
img_cropped = crop(color_img)
print(img_cropped.shape)

plt.imshow(img_cropped.permute(1, 2, 0)) # Convert from (C, H, W) (Pytorch default format) to (H, W, C) (Matplotlib format) for displaying

print("Min pixel value:", img_cropped.min().item())
print("Max pixel value:", img_cropped.max().item())
```

    torch.Size([3, 300, 300])
    Min pixel value: 0.0
    Max pixel value: 1.0
    


    
![png](Task4_PART_A_files/Task4_PART_A_3_1.png)
    



```python
def show_images(original, masked, title_mask):
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(original.permute(1,2,0).cpu().numpy())
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Show masked image with NaNs replaced by 0 (black pixels) for visualization
    axes[1].imshow(torch.nan_to_num(masked, nan=0.0).permute(1,2,0).cpu().numpy())
    axes[1].set_title("Masked")
    axes[1].axis('off')
    
    plt.suptitle(title_mask)
    plt.show()

def mask_rectangular(image, top=100, left=100, block_size=30):
    masked = image.clone()
    masked[:, top:top+block_size, left:left+block_size] = float('nan')  # put NaNs
    return masked

def mask_random(image, num_pixels=900):
    masked = image.clone()
    C, H, W = masked.shape
    
    # Pick random flat indices
    flat_indices = torch.randperm(H*W)[:num_pixels]
    
    # Convert flat indices to 2D
    rows = flat_indices // W
    cols = flat_indices % W
    
    # Mask pixels (all channels at once)
    masked[:, rows, cols] = float('nan')
    return masked


masked_rect = mask_rectangular(img_cropped, top=100, left=100)  # choose where to cut
show_images(img_cropped, masked_rect, title_mask="Masked (30*30 rectangular pixels)")

masked_rand = mask_random(img_cropped, num_pixels=900)
show_images(img_cropped, masked_rand, title_mask="Masked (900 random pixels)")

```


    
![png](Task4_PART_A_files/Task4_PART_A_4_0.png)
    



    
![png](Task4_PART_A_files/Task4_PART_A_4_1.png)
    



```python
import torch
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
import matplotlib.pyplot as plt


# -----------------------------------------------------
# 1. MATRIX FACTORIZATION USING GRADIENT DESCENT
# -----------------------------------------------------
def matrix_factorization_gradient(image_tensor, rank=20, learning_rate=0.01, num_epochs=5000, tol=1e-6, device=torch.device("cpu")):
    """
    Perform matrix factorization on masked image using gradient descent (Adam).

    Args:
        M (torch.Tensor): Masked image tensor of shape (C, H, W)
        r (int): Rank for low-rank approximation
        lr (float): Learning rate
        steps (int): Number of iterations
        tol (float): Tolerance for early stopping
        device (torch.device): CPU or CUDA device

    Returns:
        W_list (torch.Tensor): Factor matrix W (C, H, r)
        H_list (torch.Tensor): Factor matrix H (C, r, W)
        loss_list (list): Loss values per channel
    """
    image_tensor = image_tensor.to(device)
    channels, height, width = image_tensor.shape
    W = torch.randn(channels, height, rank, requires_grad=True, device=device)
    H = torch.randn(channels, rank, width, requires_grad=True, device=device)
    optimizer = optim.Adam([W, H], lr=learning_rate)
    mask = ~torch.isnan(image_tensor)
    loss_list = []
    prev_loss = float('inf')
    for i in range(1, num_epochs + 1):
        diff_matrix = torch.einsum('chr,crw->chw', W, H) - image_tensor # Reconstructed (Matrix Multiplication of W and H) - Original
        diff_vector = diff_matrix[mask] # Only consider known pixels
        loss = torch.norm(diff_vector)  # Frobenius norm = L2 Norm
        loss_list.append(loss.item())
        if i % 1000 == 0:
            print(f"Iteration {i}, loss: {loss.item()}")

        if abs(prev_loss - loss.item()) < tol:
            print(f"Converged at iteration {i}, loss: {loss.item()}")
            break
        prev_loss = loss.item()
        
        # Backpropagation and optimization step - Clear gradients, compute gradients, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return W, H, loss_list


# -----------------------------------------------------
# 2. PLOT RESULTS
# -----------------------------------------------------
def plot_result(original_img, masked_img, reconstructed_img, rank, mask_value):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Image Reconstruction with Rank={rank} and Mask={mask_value}x{mask_value} patch")

    # Original image
    axes[0].imshow(rearrange(original_img, 'c h w -> h w c').cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Masked image
    axes[1].imshow(rearrange(masked_img, 'c h w -> h w c').cpu().numpy())
    axes[1].set_title("Masked Image")
    axes[1].axis('off')

    # Reconstructed image
    axes[2].imshow(rearrange(reconstructed_img, 'c h w -> h w c').cpu().detach().numpy())
    axes[2].set_title("Reconstructed Image")
    axes[2].axis('off')

    plt.show()


# -----------------------------------------------------
# 3. RMSE + PSNR METRICS
# -----------------------------------------------------
def calculate_rmse_psnr(original_img, reconstructed_img):
    if original_img.device != reconstructed_img.device:
        original_img = original_img.to(reconstructed_img.device)

    mse = F.mse_loss(reconstructed_img, original_img)
    rmse = torch.sqrt(mse)
    psnr = 20 * torch.log10(1.0 / rmse) # Max pixel value is 1.0 since we normalized the image
    print(f"RMSE: {rmse.item():.6f}, PSNR: {psnr.item():.6f}")
    return rmse.item(), psnr.item()

# -----------------------------------------------------
# 4. FULL IMAGE RECONSTRUCTION PIPELINE
# -----------------------------------------------------
def image_reconstruction_matrix(img, masked_img, rank=20, learning_rate=1e-4, num_epochs=5000,
                                tol=1e-6, plot=True, device=torch.device("cpu")):
    img = img.to(torch.float32)
    masked_img = masked_img.to(torch.float32)

    # Perform matrix factorization
    W, H, loss_list = matrix_factorization_gradient(masked_rect, rank, learning_rate, num_epochs, tol, device)

    # Reconstruct
    reconstructed_img = torch.einsum('chr,crw->chw', W, H) # Matrix Multiplication
    reconstructed_img = torch.clamp(reconstructed_img, 0, 1) # If value > 1, set to 1, if value < 0, set to 0

    # Keep known pixels from original
    mask = ~torch.isnan(masked_img)
    reconstructed_img[mask] = img[mask]

    # Metrics
    rmse, psnr = calculate_rmse_psnr(img, reconstructed_img)

    if plot:
        plot_result(img, masked_img, reconstructed_img, rank, mask_value=30)

    return reconstructed_img, loss_list, rmse, psnr

```


```python
reconstructed_img, loss_list, rmse, psnr = image_reconstruction_matrix(img_cropped, masked_rect, rank=50, learning_rate=0.01, num_epochs=10000, tol=1e-6, plot=True, device=device)

```

    Iteration 1000, loss: 34.76765441894531
    Iteration 2000, loss: 27.726158142089844
    Iteration 3000, loss: 24.599828720092773
    Converged at iteration 3648, loss: 23.337982177734375
    RMSE: 0.007352, PSNR: 42.671894
    


    
![png](Task4_PART_A_files/Task4_PART_A_6_1.png)
    



```python
ranks = [5, 10, 20, 50, 100, 150, 200, 400]
results_patch_rect_grad = {}
reconstructed_images_patch_rect_grad = {}

for rank in ranks:
    print("-"*50)
    print(f"Evaluating rank: {rank}")
    print("-"*50)
    reconstructed_img_rect_grad, loss_list_rect_grad, rmse_rect_grad, psnr_rect_grad = image_reconstruction_matrix(
        img_cropped, masked_rect, rank, learning_rate=0.01, num_epochs=10000, plot=True, device=device
    )
    results_patch_rect_grad[rank] = {'Loss': loss_list_rect_grad[-1], 'RMSE': rmse_rect_grad, 'PSNR': psnr_rect_grad}
    reconstructed_images_patch_rect_grad[rank] = reconstructed_img_rect_grad
    print("\n\n")
```

    --------------------------------------------------
    Evaluating rank: 5
    --------------------------------------------------
    Iteration 1000, loss: 68.04456329345703
    Iteration 2000, loss: 62.640281677246094
    Converged at iteration 2566, loss: 62.42113494873047
    RMSE: 0.010956, PSNR: 39.206600
    


    
![png](Task4_PART_A_files/Task4_PART_A_7_1.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 10
    --------------------------------------------------
    Iteration 1000, loss: 58.87605667114258
    Iteration 2000, loss: 51.959068298339844
    Converged at iteration 2584, loss: 50.307029724121094
    RMSE: 0.009792, PSNR: 40.182362
    


    
![png](Task4_PART_A_files/Task4_PART_A_7_3.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 20
    --------------------------------------------------
    Iteration 1000, loss: 48.273170471191406
    Iteration 2000, loss: 40.76818084716797
    Iteration 3000, loss: 37.96436309814453
    Converged at iteration 3156, loss: 37.74274444580078
    RMSE: 0.009925, PSNR: 40.065697
    


    
![png](Task4_PART_A_files/Task4_PART_A_7_5.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 50
    --------------------------------------------------
    Iteration 1000, loss: 34.4340705871582
    Converged at iteration 1838, loss: 28.38167953491211
    RMSE: 0.008891, PSNR: 41.020580
    


    
![png](Task4_PART_A_files/Task4_PART_A_7_7.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 100
    --------------------------------------------------
    Iteration 1000, loss: 24.65315818786621
    Converged at iteration 1157, loss: 23.548919677734375
    RMSE: 0.010085, PSNR: 39.926533
    


    
![png](Task4_PART_A_files/Task4_PART_A_7_9.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 150
    --------------------------------------------------
    Iteration 1000, loss: 20.20585060119629
    Iteration 2000, loss: 16.05418586730957
    Converged at iteration 2351, loss: 15.17437744140625
    RMSE: 0.011138, PSNR: 39.063828
    


    
![png](Task4_PART_A_files/Task4_PART_A_7_11.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 200
    --------------------------------------------------
    Iteration 1000, loss: 17.08736801147461
    Converged at iteration 1531, loss: 14.968915939331055
    RMSE: 0.016415, PSNR: 35.695045
    


    
![png](Task4_PART_A_files/Task4_PART_A_7_13.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 400
    --------------------------------------------------
    Iteration 1000, loss: 12.795358657836914
    Iteration 2000, loss: 11.368173599243164
    Iteration 3000, loss: 10.707222938537598
    Iteration 4000, loss: 10.288851737976074
    Converged at iteration 4459, loss: 10.14673137664795
    RMSE: 0.050858, PSNR: 25.872862
    


    
![png](Task4_PART_A_files/Task4_PART_A_7_15.png)
    


    
    
    
    


```python
# Show original image, masked image once, and then reconstructed image for each rank
plt.figure(figsize=(8, 4))
plt.suptitle("Image Reconstruction with varying Rank")
# Plot original image
plt.subplot(1, 2, 1)
plt.imshow(rearrange(img_cropped, 'c h w -> h w c').cpu().numpy())
plt.title("Original Image")
plt.axis('off')
# Plot masked image
plt.subplot(1, 2, 2)
plt.imshow(rearrange(masked_rect, 'c h w -> h w c').cpu().numpy())
plt.title("Masked Image")
plt.axis('off')
plt.show()

plt.figure(figsize=(20, 10))
# Plot reconstructed images for each rank
for i, rank in enumerate(ranks):
    plt.subplot(2, (len(ranks)+1)//2, i+1)
    plt.imshow(rearrange(reconstructed_images_patch_rect_grad[rank], 'c h w -> h w c').cpu().detach().numpy())
    plt.title(f"Rank={rank}")
    plt.axis('off')
plt.show()
```


    
![png](Task4_PART_A_files/Task4_PART_A_8_0.png)
    



    
![png](Task4_PART_A_files/Task4_PART_A_8_1.png)
    



```python
for rank, metrics in results_patch_rect_grad.items():
    print(f"Rank: {rank}, Loss: {metrics['Loss']:.4f}, RMSE: {metrics['RMSE']:.4f}, PSNR: {metrics['PSNR']:.4f}")
ranks_list_rect_grad = list(results_patch_rect_grad.keys())
loss_list_rect_grad = [metrics['Loss'] for metrics in results_patch_rect_grad.values()]
rmse_list_rect_grad = [metrics['RMSE'] for metrics in results_patch_rect_grad.values()]
psnr_list_rect_grad = [metrics['PSNR'] for metrics in results_patch_rect_grad.values()]

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(ranks_list_rect_grad, loss_list_rect_grad, marker='o')
plt.xlabel('Rank')
plt.ylabel('Loss')
plt.title('Loss vs. Rank')
plt.subplot(1, 3, 2)
plt.plot(ranks_list_rect_grad, rmse_list_rect_grad, marker='o')
plt.xlabel('Rank')
plt.ylabel('RMSE')
plt.title('RMSE vs. Rank')
plt.subplot(1, 3, 3)
plt.plot(ranks_list_rect_grad, psnr_list_rect_grad, marker='o')
plt.xlabel('Rank')
plt.ylabel('PSNR')
plt.title('PSNR vs. Rank')
plt.show()
```

    Rank: 5, Loss: 62.4211, RMSE: 0.0110, PSNR: 39.2066
    Rank: 10, Loss: 50.3070, RMSE: 0.0098, PSNR: 40.1824
    Rank: 20, Loss: 37.7427, RMSE: 0.0099, PSNR: 40.0657
    Rank: 50, Loss: 28.3817, RMSE: 0.0089, PSNR: 41.0206
    Rank: 100, Loss: 23.5489, RMSE: 0.0101, PSNR: 39.9265
    Rank: 150, Loss: 15.1744, RMSE: 0.0111, PSNR: 39.0638
    Rank: 200, Loss: 14.9689, RMSE: 0.0164, PSNR: 35.6950
    Rank: 400, Loss: 10.1467, RMSE: 0.0509, PSNR: 25.8729
    


    
![png](Task4_PART_A_files/Task4_PART_A_9_1.png)
    



```python
best_psnr_rect_grad = max(psnr_list_rect_grad)
best_psnr_idx_rect_grad = psnr_list_rect_grad.index(best_psnr_rect_grad)
best_rank_rect_grad = ranks_list_rect_grad[best_psnr_idx_rect_grad]
best_rmse_rect_grad = rmse_list_rect_grad[best_psnr_idx_rect_grad]

print("Optimal Rank and Metrics:\n")
print(f"Rank: {best_rank_rect_grad}")
print(f"PSNR: {best_psnr_rect_grad:.4f}")
print(f"RMSE: {best_rmse_rect_grad:.4f}")
```

    Optimal Rank and Metrics:
    
    Rank: 50
    PSNR: 41.0206
    RMSE: 0.0089
    


```python
reconstructed_random_grad, loss_random_grad, rmse_random_grad, psnr_random_grad = image_reconstruction_matrix(
    img_cropped, masked_rand, rank=20, learning_rate=0.01, num_epochs=10000, tol=1e-6, plot=True, device=torch.device("cpu")
)

```

    Iteration 1000, loss: 48.427162170410156
    Iteration 2000, loss: 41.2430305480957
    Iteration 3000, loss: 38.39360809326172
    Iteration 4000, loss: 37.45574188232422
    Converged at iteration 4498, loss: 37.30405044555664
    RMSE: 0.006421, PSNR: 43.847729
    


    
![png](Task4_PART_A_files/Task4_PART_A_11_1.png)
    



```python
ranks = [5, 10, 20, 50, 100, 150, 200, 400]
results_patch_random_grad = {}
reconstructed_images_patch_random_grad = {}

for rank in ranks:
    print("-"*50)
    print(f"Evaluating rank: {rank}")
    print("-"*50)
    reconstructed_img_random_grad, loss_list_random_grad, rmse_random_grad, psnr_random_grad = image_reconstruction_matrix(
        img_cropped, masked_rand, rank, learning_rate=0.01, num_epochs=10000, plot=True, device=device
    )

    results_patch_random_grad[rank] = {'Loss': loss_list_random_grad[-1], 'RMSE': rmse_random_grad, 'PSNR': psnr_random_grad}
    reconstructed_images_patch_random_grad[rank] = reconstructed_img_random_grad
    print("\n\n")
```

    --------------------------------------------------
    Evaluating rank: 5
    --------------------------------------------------
    Iteration 1000, loss: 68.29059600830078
    Iteration 2000, loss: 62.51821517944336
    Converged at iteration 2273, loss: 62.42795944213867
    RMSE: 0.011399, PSNR: 38.862843
    


    
![png](Task4_PART_A_files/Task4_PART_A_12_1.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 10
    --------------------------------------------------
    Iteration 1000, loss: 59.44832229614258
    Iteration 2000, loss: 52.32437515258789
    Iteration 3000, loss: 50.00370788574219
    Converged at iteration 3913, loss: 49.73929977416992
    RMSE: 0.009044, PSNR: 40.873108
    


    
![png](Task4_PART_A_files/Task4_PART_A_12_3.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 20
    --------------------------------------------------
    Iteration 1000, loss: 48.95173263549805
    Iteration 2000, loss: 41.183876037597656
    Iteration 3000, loss: 38.252986907958984
    Converged at iteration 3381, loss: 37.758262634277344
    RMSE: 0.006735, PSNR: 43.433395
    


    
![png](Task4_PART_A_files/Task4_PART_A_12_5.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 50
    --------------------------------------------------
    Iteration 1000, loss: 34.59613037109375
    Iteration 2000, loss: 27.566774368286133
    Iteration 3000, loss: 24.390289306640625
    Iteration 4000, loss: 22.549152374267578
    Iteration 5000, loss: 21.462804794311523
    Converged at iteration 5108, loss: 21.338109970092773
    RMSE: 0.003964, PSNR: 48.037132
    


    
![png](Task4_PART_A_files/Task4_PART_A_12_7.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 100
    --------------------------------------------------
    Iteration 1000, loss: 24.811725616455078
    Iteration 2000, loss: 19.58082389831543
    Iteration 3000, loss: 17.104040145874023
    Iteration 4000, loss: 15.5360746383667
    Iteration 5000, loss: 14.4513521194458
    Iteration 6000, loss: 13.596306800842285
    Iteration 7000, loss: 12.932978630065918
    Iteration 8000, loss: 12.394889831542969
    Iteration 9000, loss: 11.932838439941406
    Iteration 10000, loss: 11.518310546875
    RMSE: 0.002409, PSNR: 52.362099
    


    
![png](Task4_PART_A_files/Task4_PART_A_12_9.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 150
    --------------------------------------------------
    Iteration 1000, loss: 19.934751510620117
    Iteration 2000, loss: 15.88932991027832
    Iteration 3000, loss: 13.882695198059082
    Iteration 4000, loss: 12.613967895507812
    Converged at iteration 4340, loss: 12.273750305175781
    RMSE: 0.002741, PSNR: 51.241535
    


    
![png](Task4_PART_A_files/Task4_PART_A_12_11.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 200
    --------------------------------------------------
    Iteration 1000, loss: 17.191930770874023
    Iteration 2000, loss: 13.851380348205566
    Iteration 3000, loss: 12.160633087158203
    Converged at iteration 3459, loss: 11.637883186340332
    RMSE: 0.002800, PSNR: 51.055878
    


    
![png](Task4_PART_A_files/Task4_PART_A_12_13.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 400
    --------------------------------------------------
    Iteration 1000, loss: 12.710806846618652
    Iteration 2000, loss: 11.346916198730469
    Iteration 3000, loss: 10.690279006958008
    Converged at iteration 3846, loss: 10.339591026306152
    RMSE: 0.006225, PSNR: 44.117332
    


    
![png](Task4_PART_A_files/Task4_PART_A_12_15.png)
    


    
    
    
    


```python

# Show original image, masked image once, and then reconstructed image for each rank
plt.figure(figsize=(8, 4))
plt.suptitle("Image Reconstruction with varying Rank")
# Plot original image
plt.subplot(1, 2, 1)
plt.imshow(rearrange(img_cropped, 'c h w -> h w c').cpu().numpy())
plt.title("Original Image")
plt.axis('off')
# Plot masked image
plt.subplot(1, 2, 2)
plt.imshow(rearrange(masked_rand, 'c h w -> h w c').cpu().numpy())
plt.title("Masked Image")
plt.axis('off')
plt.show()

plt.figure(figsize=(20, 10))
# Plot reconstructed images for each rank
for i, rank in enumerate(ranks):
    plt.subplot(2, (len(ranks)+1)//2, i+1)
    plt.imshow(rearrange(reconstructed_images_patch_random_grad[rank], 'c h w -> h w c').cpu().detach().numpy())
    plt.title(f"Rank={rank}")
    plt.axis('off')
plt.show()
```


    
![png](Task4_PART_A_files/Task4_PART_A_13_0.png)
    



    
![png](Task4_PART_A_files/Task4_PART_A_13_1.png)
    



```python
for rank, metrics in results_patch_random_grad.items():
    print(f"Rank: {rank}, Loss: {metrics['Loss']:.4f}, RMSE: {metrics['RMSE']:.4f}, PSNR: {metrics['PSNR']:.4f}")
ranks_list_random_grad = list(results_patch_random_grad.keys())
loss_list_random_grad = [metrics['Loss'] for metrics in results_patch_random_grad.values()]
rmse_list_random_grad = [metrics['RMSE'] for metrics in results_patch_random_grad.values()]
psnr_list_random_grad = [metrics['PSNR'] for metrics in results_patch_random_grad.values()]

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(ranks_list_random_grad, loss_list_random_grad, marker='o')
plt.xlabel('Rank')
plt.ylabel('Loss')
plt.title('Loss vs. Rank')
plt.subplot(1, 3, 2)
plt.plot(ranks_list_random_grad, rmse_list_random_grad, marker='o')
plt.xlabel('Rank')
plt.ylabel('RMSE')
plt.title('RMSE vs. Rank')
plt.subplot(1, 3, 3)
plt.plot(ranks_list_random_grad, psnr_list_random_grad, marker='o')
plt.xlabel('Rank')
plt.ylabel('PSNR')
plt.title('PSNR vs. Rank')
plt.show()
```

    Rank: 5, Loss: 62.4280, RMSE: 0.0114, PSNR: 38.8628
    Rank: 10, Loss: 49.7393, RMSE: 0.0090, PSNR: 40.8731
    Rank: 20, Loss: 37.7583, RMSE: 0.0067, PSNR: 43.4334
    Rank: 50, Loss: 21.3381, RMSE: 0.0040, PSNR: 48.0371
    Rank: 100, Loss: 11.5183, RMSE: 0.0024, PSNR: 52.3621
    Rank: 150, Loss: 12.2738, RMSE: 0.0027, PSNR: 51.2415
    Rank: 200, Loss: 11.6379, RMSE: 0.0028, PSNR: 51.0559
    Rank: 400, Loss: 10.3396, RMSE: 0.0062, PSNR: 44.1173
    


    
![png](Task4_PART_A_files/Task4_PART_A_14_1.png)
    



```python
best_psnr_random_grad = max(psnr_list_random_grad)
best_psnr_idx_random_grad = psnr_list_random_grad.index(best_psnr_random_grad)
best_rank_random_grad = ranks_list_random_grad[best_psnr_idx_random_grad]
best_rmse_random_grad = rmse_list_random_grad[best_psnr_idx_random_grad]

print("Optimal Rank and Metrics:\n")
print(f"Rank: {best_rank_random_grad}")
print(f"PSNR: {best_psnr_random_grad:.4f}")
print(f"RMSE: {best_rmse_random_grad:.4f}")
```

    Optimal Rank and Metrics:
    
    Rank: 100
    PSNR: 52.3621
    RMSE: 0.0024
    


```python
import torch
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. MATRIX FACTORIZATION USING ALS
# -----------------------------------------------------
def matrix_factorization_als(M, r=20, steps=100, tol=1e-5, device=torch.device("cpu")):
    C, H, W = M.shape
    M = M.to(device)
    
    W_list = torch.empty(C, H, r, device=device)
    H_list = torch.empty(C, r, W, device=device)
    loss_list = [[] for _ in range(C)]

    for c in range(C):
        mask = ~torch.isnan(M[c])
        M_channel = M[c].clone()
        M_channel[~mask] = 0.0

        U = torch.rand(H, r, device=device)
        V = torch.rand(W, r, device=device)

        prev_loss = float('inf')

        for step in range(1, steps + 1):
            # --- Update U ---
            for i in range(H): # For each row
                cols = mask[i] # Known pixels in this row
                if cols.sum() == 0: # If no known pixels, then skip this row
                    continue
                V_sub = V[cols] # Select only the columns with known pixels
                M_sub = M_channel[i, cols] # take corresponding known pixels
                sol = torch.linalg.lstsq(V_sub, M_sub.unsqueeze(1)).solution.squeeze() # Solve Ax = b using least squares
                U[i] = sol

            # --- Update V ---
            for j in range(W):
                rows = mask[:, j]
                if rows.sum() == 0:
                    continue
                U_sub = U[rows]
                M_sub = M_channel[rows, j]
                sol = torch.linalg.lstsq(U_sub, M_sub.unsqueeze(1)).solution.squeeze()
                V[j] = sol

            # --- Compute loss ---
            M_pred = U @ V.T
            error = (M_channel - M_pred)[mask]
            loss = torch.sqrt((error ** 2).mean())
            loss_list[c].append(loss.item())

            if step % 10 == 0:
                print(f"Channel {c}, Step {step}, RMSE={loss.item():.6f}")

            if abs(prev_loss - loss.item()) < tol:
                print(f"Channel {c} converged at step {step}, RMSE={loss.item():.6f}")
                break
            prev_loss = loss.item()

        W_list[c] = U
        H_list[c] = V.T

    return W_list, H_list, loss_list


# -----------------------------------------------------
# 2. METRICS AND VISUALIZATION
# -----------------------------------------------------
def calculate_rmse_psnr(original_img, reconstructed_img):
    mse = F.mse_loss(reconstructed_img, original_img)
    rmse = torch.sqrt(mse)
    psnr = 20 * torch.log10(1.0 / rmse)
    print(f"RMSE: {rmse.item():.6f}, PSNR: {psnr.item():.6f}")
    return rmse.item(), psnr.item()


def plot_result(original_img, masked_img, reconstructed_img, rank, mask_value):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Image Reconstruction with Rank={rank} and Mask={mask_value}x{mask_value}")
    axes[0].imshow(rearrange(original_img, 'c h w -> h w c').cpu().numpy()); axes[0].set_title("Original")
    axes[1].imshow(rearrange(masked_img, 'c h w -> h w c').cpu().numpy()); axes[1].set_title("Masked")
    axes[2].imshow(rearrange(reconstructed_img, 'c h w -> h w c').cpu().detach().numpy()); axes[2].set_title("Reconstructed")
    for ax in axes: ax.axis('off')
    plt.show()


# -----------------------------------------------------
# 3. FULL IMAGE RECONSTRUCTION PIPELINE
# -----------------------------------------------------
def image_reconstruction_matrix_als(img, masked_img, rank=20, tol=1e-5, plot=True, device=torch.device("cpu")):
    img = img.to(torch.float32)
    masked_img = masked_img.to(torch.float32)

    # Perform ALS factorization
    W, H, loss_list = matrix_factorization_als(masked_img, r=rank, steps=100, tol=tol, device=device)

    # Reconstruction
    reconstructed_img = torch.einsum('chr,crw->chw', W, H)
    reconstructed_img = torch.clamp(reconstructed_img, 0, 1)

    mask = ~torch.isnan(masked_img)
    reconstructed_img[mask] = img[mask]

    rmse, psnr = calculate_rmse_psnr(img, reconstructed_img)

    if plot:
        plot_result(img, masked_img, reconstructed_img, rank, mask_value=30)

    return reconstructed_img, loss_list, rmse, psnr

```


```python
# Rectangular block reconstruction
reconstructed_rect_als, loss_rect_als, rmse_rect_als, psnr_rect_als = image_reconstruction_matrix_als(
    img_cropped, masked_rect, rank=20, plot=True
)

# # Random pixels reconstruction
# reconstructed_rand_als, loss_rand_als, rmse_rand_als, psnr_rand_als = image_reconstruction_matrix(
#     img_cropped, masked_rand, rank=20, plot=True
# )


```

    Channel 0, Step 10, RMSE=0.070397
    Channel 0 converged at step 13, RMSE=0.070357
    Channel 1, Step 10, RMSE=0.071619
    Channel 1 converged at step 19, RMSE=0.071506
    Channel 2, Step 10, RMSE=0.072655
    Channel 2 converged at step 18, RMSE=0.072505
    RMSE: 0.009913, PSNR: 40.076271
    


    
![png](Task4_PART_A_files/Task4_PART_A_17_1.png)
    



```python
ranks = [5, 10, 20, 50, 100, 150, 200, 400]
results_patch_rect_als = {}
reconstructed_images_patch_rect_als = {}

for rank in ranks:
    print("-"*50)
    print(f"Evaluating rank: {rank}")
    print("-"*50)
    reconstructed_img_rect_als, loss_list_rect_als, rmse_rect_als, psnr_rect_als = image_reconstruction_matrix_als(
    img_cropped, masked_rect, rank, plot=True
)

    results_patch_rect_als[rank] = {'Loss': loss_list_rect_als[-1], 'RMSE': rmse_rect_als, 'PSNR': psnr_rect_als}
    reconstructed_images_patch_rect_als[rank] = reconstructed_img_rect_als
    print("\n\n")




```

    --------------------------------------------------
    Evaluating rank: 5
    --------------------------------------------------
    Channel 0 converged at step 7, RMSE=0.116243
    Channel 1, Step 10, RMSE=0.119212
    Channel 1 converged at step 13, RMSE=0.119171
    Channel 2 converged at step 8, RMSE=0.126419
    RMSE: 0.010949, PSNR: 39.212738
    


    
![png](Task4_PART_A_files/Task4_PART_A_18_1.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 10
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.093298
    Channel 0 converged at step 15, RMSE=0.093182
    Channel 1, Step 10, RMSE=0.095486
    Channel 1 converged at step 13, RMSE=0.095452
    Channel 2, Step 10, RMSE=0.099707
    Channel 2, Step 20, RMSE=0.099486
    Channel 2 converged at step 22, RMSE=0.099466
    RMSE: 0.010324, PSNR: 39.723431
    


    
![png](Task4_PART_A_files/Task4_PART_A_18_3.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 20
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.070525
    Channel 0 converged at step 17, RMSE=0.070373
    Channel 1, Step 10, RMSE=0.071654
    Channel 1, Step 20, RMSE=0.071485
    Channel 1 converged at step 20, RMSE=0.071485
    Channel 2, Step 10, RMSE=0.072590
    Channel 2 converged at step 15, RMSE=0.072483
    RMSE: 0.009095, PSNR: 40.823788
    


    
![png](Task4_PART_A_files/Task4_PART_A_18_5.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 50
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.037988
    Channel 0 converged at step 12, RMSE=0.037965
    Channel 1, Step 10, RMSE=0.038030
    Channel 1 converged at step 15, RMSE=0.037952
    Channel 2, Step 10, RMSE=0.036770
    Channel 2 converged at step 12, RMSE=0.036746
    RMSE: 0.008018, PSNR: 41.918365
    


    
![png](Task4_PART_A_files/Task4_PART_A_18_7.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 100
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.015746
    Channel 0 converged at step 11, RMSE=0.015739
    Channel 1, Step 10, RMSE=0.015711
    Channel 1 converged at step 11, RMSE=0.015701
    Channel 2, Step 10, RMSE=0.015377
    Channel 2 converged at step 11, RMSE=0.015368
    RMSE: 0.011228, PSNR: 38.993744
    


    
![png](Task4_PART_A_files/Task4_PART_A_18_9.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 150
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.006738
    Channel 0 converged at step 11, RMSE=0.006730
    Channel 1, Step 10, RMSE=0.006690
    Channel 1 converged at step 10, RMSE=0.006690
    Channel 2, Step 10, RMSE=0.006732
    Channel 2 converged at step 10, RMSE=0.006732
    RMSE: 0.020525, PSNR: 33.754536
    


    
![png](Task4_PART_A_files/Task4_PART_A_18_11.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 200
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.002413
    Channel 0 converged at step 10, RMSE=0.002413
    Channel 1, Step 10, RMSE=0.002390
    Channel 1 converged at step 10, RMSE=0.002390
    Channel 2, Step 10, RMSE=0.002440
    Channel 2 converged at step 11, RMSE=0.002431
    RMSE: 0.028869, PSNR: 30.791416
    


    
![png](Task4_PART_A_files/Task4_PART_A_18_13.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 400
    --------------------------------------------------
    Channel 0 converged at step 4, RMSE=0.000003
    Channel 1 converged at step 2, RMSE=0.000072
    Channel 2 converged at step 8, RMSE=0.000003
    RMSE: 0.019744, PSNR: 34.091251
    


    
![png](Task4_PART_A_files/Task4_PART_A_18_15.png)
    


    
    
    
    


```python

# Show original image, masked image once, and then reconstructed image for each rank
plt.figure(figsize=(8, 4))
plt.suptitle("Image Reconstruction with varying Rank")
# Plot original image
plt.subplot(1, 2, 1)
plt.imshow(rearrange(img_cropped, 'c h w -> h w c').cpu().numpy())
plt.title("Original Image")
plt.axis('off')
# Plot masked image
plt.subplot(1, 2, 2)
plt.imshow(rearrange(masked_rect, 'c h w -> h w c').cpu().numpy())
plt.title("Masked Image")
plt.axis('off')
plt.show()

plt.figure(figsize=(20, 10))
# Plot reconstructed images for each rank
for i, rank in enumerate(ranks):
    plt.subplot(2, (len(ranks)+1)//2, i+1)
    plt.imshow(rearrange(reconstructed_images_patch_rect_als[rank], 'c h w -> h w c').cpu().detach().numpy())
    plt.title(f"Rank={rank}")
    plt.axis('off')
plt.show()
```


    
![png](Task4_PART_A_files/Task4_PART_A_19_0.png)
    



    
![png](Task4_PART_A_files/Task4_PART_A_19_1.png)
    



```python
for rank_als, metrics_als in results_patch_rect_als.items():
    final_loss_als = metrics_als['Loss'][-1] if isinstance(metrics_als['Loss'], list) else metrics_als['Loss']
    print(f"Rank: {rank_als}, Loss: {final_loss_als:.4f}, RMSE: {metrics_als['RMSE']:.4f}, PSNR: {metrics_als['PSNR']:.4f}")

ranks_list_rect_als = list(results_patch_rect_als.keys())
loss_list_rect_als = [
    metrics['Loss'][-1] if isinstance(metrics['Loss'], list) else metrics['Loss']
    for metrics in results_patch_rect_als.values()
]
rmse_list_rect_als = [metrics['RMSE'] for metrics in results_patch_rect_als.values()]
psnr_list_rect_als = [metrics['PSNR'] for metrics in results_patch_rect_als.values()]

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(ranks_list_rect_als, loss_list_rect_als, marker='o')
plt.xlabel('Rank')
plt.ylabel('Loss')
plt.title('Loss vs. Rank')

plt.subplot(1, 3, 2)
plt.plot(ranks_list_rect_als, rmse_list_rect_als, marker='o')
plt.xlabel('Rank')
plt.ylabel('RMSE')
plt.title('RMSE vs. Rank')

plt.subplot(1, 3, 3)
plt.plot(ranks_list_rect_als, psnr_list_rect_als, marker='o')
plt.xlabel('Rank')
plt.ylabel('PSNR')
plt.title('PSNR vs. Rank')

plt.tight_layout()
plt.show()

```

    Rank: 5, Loss: 0.0725, RMSE: 0.0096, PSNR: 40.3754
    Rank: 10, Loss: 0.0725, RMSE: 0.0088, PSNR: 41.1574
    Rank: 20, Loss: 0.0725, RMSE: 0.0092, PSNR: 40.7630
    Rank: 50, Loss: 0.0725, RMSE: 0.0093, PSNR: 40.6016
    Rank: 100, Loss: 0.0725, RMSE: 0.0099, PSNR: 40.0709
    Rank: 150, Loss: 0.0725, RMSE: 0.0101, PSNR: 39.9340
    Rank: 200, Loss: 0.0725, RMSE: 0.0091, PSNR: 40.8613
    Rank: 400, Loss: 0.0725, RMSE: 0.0090, PSNR: 40.9013
    


    
![png](Task4_PART_A_files/Task4_PART_A_20_1.png)
    



```python
best_psnr_rect_als = max(psnr_list_rect_als)
best_psnr_idx_rect_als = psnr_list_rect_als.index(best_psnr_rect_als)
best_rank_rect_als = ranks_list_rect_als[best_psnr_idx_rect_als]
best_rmse_rect_als = rmse_list_rect_als[best_psnr_idx_rect_als]

print("Optimal Rank and Metrics:\n")
print(f"Rank: {best_rank_rect_als}")
print(f"PSNR: {best_psnr_rect_als:.4f}")
print(f"RMSE: {best_rmse_rect_als:.4f}")
```

    Optimal Rank and Metrics:
    
    Rank: 10
    PSNR: 41.1574
    RMSE: 0.0088
    


```python
# Random pixels reconstruction
reconstructed_rand_als, loss_rand_als, rmse_rand_als, psnr_rand_als = image_reconstruction_matrix_als(
    img_cropped, masked_rand, rank=20, plot=True
)


```

    Channel 0, Step 10, RMSE=0.070403
    Channel 0, Step 20, RMSE=0.070229
    Channel 0 converged at step 21, RMSE=0.070220
    Channel 1, Step 10, RMSE=0.071469
    Channel 1 converged at step 16, RMSE=0.071365
    Channel 2, Step 10, RMSE=0.072375
    Channel 2 converged at step 13, RMSE=0.072324
    RMSE: 0.007654, PSNR: 42.322502
    


    
![png](Task4_PART_A_files/Task4_PART_A_22_1.png)
    



```python
ranks = [5, 10, 20, 50, 100, 150, 200, 400]
results_patch_rand_als = {}
reconstructed_images_patch_rand_als = {}

for rank in ranks:
    print("-"*50)
    print(f"Evaluating rank: {rank}")
    print("-"*50)
    reconstructed_img_rand_als, loss_list_rand_als, rmse_rand_als, psnr_rand_als = image_reconstruction_matrix_als(
        img_cropped, masked_rand, rank, plot=True
    )
    

    results_patch_rand_als[rank] = {'Loss': loss_list_rand_als[-1], 'RMSE': rmse_rand_als, 'PSNR': psnr_rand_als}
    reconstructed_images_patch_rand_als[rank] = reconstructed_img_rand_als
    print("\n\n")
```

    --------------------------------------------------
    Evaluating rank: 5
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.116141
    Channel 0 converged at step 12, RMSE=0.116119
    Channel 1 converged at step 7, RMSE=0.118972
    Channel 2, Step 10, RMSE=0.126536
    Channel 2 converged at step 15, RMSE=0.126223
    RMSE: 0.011804, PSNR: 38.559517
    


    
![png](Task4_PART_A_files/Task4_PART_A_23_1.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 10
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.093546
    Channel 0, Step 20, RMSE=0.093179
    Channel 0 converged at step 28, RMSE=0.093086
    Channel 1, Step 10, RMSE=0.095442
    Channel 1, Step 20, RMSE=0.095207
    Channel 1, Step 30, RMSE=0.094984
    Channel 1 converged at step 33, RMSE=0.094952
    Channel 2, Step 10, RMSE=0.099549
    Channel 2 converged at step 15, RMSE=0.099489
    RMSE: 0.009752, PSNR: 40.218437
    


    
![png](Task4_PART_A_files/Task4_PART_A_23_3.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 20
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.070216
    Channel 0 converged at step 11, RMSE=0.070207
    Channel 1, Step 10, RMSE=0.071465
    Channel 1 converged at step 15, RMSE=0.071381
    Channel 2, Step 10, RMSE=0.072385
    Channel 2 converged at step 14, RMSE=0.072326
    RMSE: 0.007756, PSNR: 42.207382
    


    
![png](Task4_PART_A_files/Task4_PART_A_23_5.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 50
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.037857
    Channel 0 converged at step 14, RMSE=0.037808
    Channel 1, Step 10, RMSE=0.037825
    Channel 1 converged at step 12, RMSE=0.037808
    Channel 2, Step 10, RMSE=0.036696
    Channel 2 converged at step 15, RMSE=0.036617
    RMSE: 0.006394, PSNR: 43.884575
    


    
![png](Task4_PART_A_files/Task4_PART_A_23_7.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 100
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.015622
    Channel 0 converged at step 11, RMSE=0.015615
    Channel 1, Step 10, RMSE=0.015620
    Channel 1 converged at step 11, RMSE=0.015611
    Channel 2, Step 10, RMSE=0.015264
    Channel 2 converged at step 10, RMSE=0.015264
    RMSE: 0.006438, PSNR: 43.825230
    


    
![png](Task4_PART_A_files/Task4_PART_A_23_9.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 150
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.006683
    Channel 0 converged at step 11, RMSE=0.006673
    Channel 1, Step 10, RMSE=0.006686
    Channel 1 converged at step 13, RMSE=0.006650
    Channel 2, Step 10, RMSE=0.006709
    Channel 2 converged at step 12, RMSE=0.006688
    RMSE: 0.008364, PSNR: 41.551956
    


    
![png](Task4_PART_A_files/Task4_PART_A_23_11.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 200
    --------------------------------------------------
    Channel 0, Step 10, RMSE=0.002463
    Channel 0 converged at step 17, RMSE=0.002353
    Channel 1, Step 10, RMSE=0.002401
    Channel 1 converged at step 15, RMSE=0.002336
    Channel 2, Step 10, RMSE=0.002511
    Channel 2 converged at step 17, RMSE=0.002401
    RMSE: 0.014817, PSNR: 36.584980
    


    
![png](Task4_PART_A_files/Task4_PART_A_23_13.png)
    


    
    
    
    --------------------------------------------------
    Evaluating rank: 400
    --------------------------------------------------
    Channel 0 converged at step 3, RMSE=0.000089
    Channel 1, Step 10, RMSE=0.000114
    Channel 1 converged at step 12, RMSE=0.000063
    Channel 2 converged at step 6, RMSE=0.000102
    RMSE: 0.029158, PSNR: 30.704971
    


    
![png](Task4_PART_A_files/Task4_PART_A_23_15.png)
    


    
    
    
    


```python

# Show original image, masked image once, and then reconstructed image for each rank
plt.figure(figsize=(8, 4))
plt.suptitle("Image Reconstruction with varying Rank")
# Plot original image
plt.subplot(1, 2, 1)
plt.imshow(rearrange(img_cropped, 'c h w -> h w c').cpu().numpy())
plt.title("Original Image")
plt.axis('off')
# Plot masked image
plt.subplot(1, 2, 2)
plt.imshow(rearrange(masked_rand, 'c h w -> h w c').cpu().numpy())
plt.title("Masked Image")
plt.axis('off')
plt.show()

plt.figure(figsize=(20, 10))
# Plot reconstructed images for each rank
for i, rank in enumerate(ranks):
    plt.subplot(2, (len(ranks)+1)//2, i+1)
    plt.imshow(rearrange(reconstructed_images_patch_rand_als[rank], 'c h w -> h w c').cpu().detach().numpy())
    plt.title(f"Rank={rank}")
    plt.axis('off')
plt.show()
```


    
![png](Task4_PART_A_files/Task4_PART_A_24_0.png)
    



    
![png](Task4_PART_A_files/Task4_PART_A_24_1.png)
    



```python
for rank_als_random, metrics_als_random in results_patch_rand_als.items():
    final_loss_rand_als = metrics_als_random['Loss'][-1] if isinstance(metrics_als_random['Loss'], list) else metrics_als_random['Loss']
    print(f"Rank: {rank_als_random}, Loss: {final_loss_rand_als:.4f}, RMSE: {metrics_als_random['RMSE']:.4f}, PSNR: {metrics_als_random['PSNR']:.4f}")

ranks_list_rand_als = list(results_patch_rand_als.keys())
loss_list_rand_als = [
    metrics['Loss'][-1] if isinstance(metrics['Loss'], list) else metrics['Loss']
    for metrics in results_patch_rand_als.values()
]
rmse_list_rand_als = [metrics['RMSE'] for metrics in results_patch_rand_als.values()]
psnr_list_rand_als = [metrics['PSNR'] for metrics in results_patch_rand_als.values()]

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(ranks_list_rand_als, loss_list_rand_als, marker='o')
plt.xlabel('Rank')
plt.ylabel('Loss')
plt.title('Loss vs. Rank')

plt.subplot(1, 3, 2)
plt.plot(ranks_list_rand_als, rmse_list_rand_als, marker='o')
plt.xlabel('Rank')
plt.ylabel('RMSE')
plt.title('RMSE vs. Rank')

plt.subplot(1, 3, 3)
plt.plot(ranks_list_rand_als, psnr_list_rand_als, marker='o')
plt.xlabel('Rank')
plt.ylabel('PSNR')
plt.title('PSNR vs. Rank')

plt.tight_layout()
plt.show()

```

    Rank: 5, Loss: 0.1262, RMSE: 0.0118, PSNR: 38.5595
    Rank: 10, Loss: 0.0995, RMSE: 0.0098, PSNR: 40.2184
    Rank: 20, Loss: 0.0723, RMSE: 0.0078, PSNR: 42.2074
    Rank: 50, Loss: 0.0366, RMSE: 0.0064, PSNR: 43.8846
    Rank: 100, Loss: 0.0153, RMSE: 0.0064, PSNR: 43.8252
    Rank: 150, Loss: 0.0067, RMSE: 0.0084, PSNR: 41.5520
    Rank: 200, Loss: 0.0024, RMSE: 0.0148, PSNR: 36.5850
    Rank: 400, Loss: 0.0001, RMSE: 0.0292, PSNR: 30.7050
    


    
![png](Task4_PART_A_files/Task4_PART_A_25_1.png)
    



```python
best_psnr_rand_als = max(psnr_list_rand_als)
best_psnr_idx_rand_als = psnr_list_rand_als.index(best_psnr_rand_als)
best_rank_rand_als = ranks_list_rand_als[best_psnr_idx_rand_als]
best_rmse_rand_als = rmse_list_rand_als[best_psnr_idx_rand_als]

print("Optimal Rank and Metrics:\n")
print(f"Rank: {best_rank_rand_als}")
print(f"PSNR: {best_psnr_rand_als:.4f}")
print(f"RMSE: {best_rmse_rand_als:.4f}")
```

    Optimal Rank and Metrics:
    
    Rank: 50
    PSNR: 43.8846
    RMSE: 0.0064
    
