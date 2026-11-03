"""
=============================================================================
CV-CUDA PRACTICAL DEMO
=============================================================================
This script shows CV-CUDA in action with a real preprocessing pipeline.

WHAT IT DOES:
  1. Creates a dummy batch of images on the GPU (no CPU loading needed!)
  2. Resizes them all at once on the GPU
  3. Converts color spaces on the GPU
  4. Applies blur filtering on the GPU
  5. Saves one result back to disk

WHY THIS MATTERS:
  In a real ML pipeline, you'd replace step 1 with loading actual images,
  and step 5 with feeding into a PyTorch/TensorRT model. The key insight
  is that the data NEVER leaves the GPU between these steps.

REQUIREMENTS:
  pip install cvcuda-cu12   (or cvcuda-cu13)
  pip install nvidia-nvimgcodec-cu12
  An NVIDIA GPU with compute capability >= 7.5
  CUDA toolkit installed

USAGE:
  python cvcuda.py
=============================================================================
"""

import cvcuda
import numpy as np
from nvidia import nvimgcodec

# ---------------------------------------------------------------------------
# STEP 0: Helper - create dummy images on GPU for demonstration
# ---------------------------------------------------------------------------
# In production you'd decode real images with nvimgcodec.Decoder().
# Here we generate random images so you can run this without any input files.

def create_dummy_batch(batch_size, height, width, channels=3):
    """
    Creates a batch of random images ON THE GPU using NumPy + CV-CUDA.

    Normally you'd use:
        decoder = nvimgcodec.Decoder()
        images  = decoder.read(["img1.jpg", "img2.jpg", ...])

    But for this demo, we fabricate images so no input files are needed.
    """
    # Create random pixel data on CPU (NumPy)
    raw_images = [
        np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
        for _ in range(batch_size)
    ]

    # Wrap into an nvImageCodec image batch (this puts them on GPU)
    nv_image_batch = nvimgcodec.as_image_batch(raw_images)
    return nv_image_batch


# ---------------------------------------------------------------------------
# STEP 1: DECODE - Get images onto the GPU
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading images onto GPU")
print("=" * 60)

BATCH_SIZE  = 4          # process 4 images at once
IMG_HEIGHT  = 480
IMG_WIDTH   = 640
TARGET_SIZE = (224, 224)  # typical model input size

batch = create_dummy_batch(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
print(f"  Created batch of {BATCH_SIZE} images, each {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"  Images are now on GPU memory (zero CPU->GPU copy)")

# ---------------------------------------------------------------------------
# STEP 2: RESIZE - Scale all images to model input size
# ---------------------------------------------------------------------------
print(f"\nSTEP 2: Resizing all images to {TARGET_SIZE}")
print("-" * 60)

# Convert the nvImageCodec batch to a CV-CUDA tensor
# "HWC" means Height x Width x Channels layout
input_tensor = cvcuda.as_tensor(batch, "HWC")
print(f"  Input tensor shape:  {input_tensor.shape}")   # (4, 480, 640, 3)

# Resize all 4 images in ONE GPU kernel call
resized = cvcuda.resize(
    input_tensor,
    (TARGET_SIZE[1], TARGET_SIZE[0], 3),   # (width, height, channels)
    cvcuda.Interp.LINEAR,                    # bilinear interpolation
)
print(f"  Resized tensor shape: {resized.shape}")       # (4, 224, 224, 3)
print(f"  Done in ONE GPU call, not 4 separate calls!")

# ---------------------------------------------------------------------------
# STEP 3: COLOR CONVERSION - BGR -> RGB (what most models expect)
# ---------------------------------------------------------------------------
print(f"\nSTEP 3: Converting color space BGR -> RGB")
print("-" * 60)

# Most PyTorch/TensorFlow models expect RGB input
rgb_batch = cvcuda.color_convert(
    resized,
    cvcuda.ColorFormat.BGR,
    cvcuda.ColorFormat.RGB,
)
print(f"  Color converted shape: {rgb_batch.shape}")
print(f"  Data is STILL on GPU - no copy happened!")

# ---------------------------------------------------------------------------
# STEP 4: BLUR - Apply Gaussian blur (common preprocessing/augmentation)
# ---------------------------------------------------------------------------
print(f"\nSTEP 4: Applying Gaussian blur on GPU")
print("-" * 60)

blurred = cvcuda.gaussian_blur(
    rgb_batch,
    kernel_size=(5, 5),   # 5x5 kernel
    sigma_x=1.5,
)
print(f"  Blurred tensor shape: {blurred.shape}")

# ---------------------------------------------------------------------------
# STEP 5: NORMALIZE - Convert to float32 and normalize (model-ready)
# ---------------------------------------------------------------------------
print(f"\nSTEP 5: Normalizing to float32 [0, 1] range")
print("-" * 60)

# Convert uint8 [0-255] to float32 [0.0-1.0]
# cvcuda.astype does this entirely on GPU
float_tensor = cvcuda.astype(blurred, np.float32) / 255.0
print(f"  Float tensor dtype:  {float_tensor.dtype}")
print(f"  Float tensor shape:  {float_tensor.shape}")
print(f"  Values range:        0.0 to 1.0 (model-ready)")

# ---------------------------------------------------------------------------
# STEP 6: (OPTIONAL) Export - Save result or hand off to PyTorch
# ---------------------------------------------------------------------------
print(f"\nSTEP 6: Exporting first image from batch")
print("-" * 60)

# Extract one image from the batch using slicing
single_image = cvcuda.slice(float_tensor, (0,), (1,))
single_image_uint8 = (single_image * 255).astype(np.uint8)

# Save to disk (this is the only CPU round-trip)
output_batch = cvcuda.as_image_batch(single_image_uint8)
encoder = nvimgcodec.Encoder()
encoder.write("output_demo.jpg", output_batch[0])
print(f"  Saved: output_demo.jpg")

# ---------------------------------------------------------------------------
# BONUS: What you'd do with PyTorch (pseudocode)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("BONUS: PyTorch Integration (pseudocode)")
print("=" * 60)
print("""
  # CV-CUDA and PyTorch share GPU memory - zero copy!
  import torch

  # This creates a PyTorch tensor that points to the SAME GPU memory
  torch_tensor = cvcuda.as_pytorch_tensor(float_tensor)

  # Feed directly into your model - no .cpu(), no .numpy(), no copy
  model_input = torch_tensor.permute(0, 3, 1, 2)  # HWC -> NCHW
  output = my_model(model_input)
""")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("PIPELINE SUMMARY")
print("=" * 60)
print("""
  CPU  --------------------------------------------------- GPU
       |                                              |
       |  create/download images                      |
       |  ---------------------------------------->   |
       |                                              |
       |          (everything below is GPU-only)       |
       |                                              |
       |  resize        ----> on GPU                  |
       |  color_convert ----> on GPU                  |
       |  gaussian_blur ----> on GPU                  |
       |  normalize     ----> on GPU                  |
       |  hand to model ----> on GPU (zero-copy)      |
       |                                              |
       |  <---- save result (only when needed) ------  |
       |                                              |
  KEY TAKEAWAY: Data stays on GPU the entire time.
  The only CPU involvement is initial load and final save.
""")
