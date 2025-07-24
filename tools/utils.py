import numpy as np

def decode_task_description(task_tensor):
    """Decode task description tensor back to string"""
    # Remove padding (zeros) and convert back to characters
    chars = [chr(c.item()) for c in task_tensor if c.item() != 0]
    return ''.join(chars)

def is_image_corrupted(img: np.ndarray, debug=False, use_fft=True) -> bool:
    """
    Heuristically detect if an image in [0, 1] float format is corrupted.
    img: (H, W, C) float32 np.array
    Returns True if image is likely corrupted.
    """
    if img.dtype != np.float32 and img.dtype != np.float64:
        raise ValueError("Image must be float32 or float64 in [0, 1] range")

    if img.min() < 0 or img.max() > 1:
        return True  # Out of expected range

    if use_fft:
        import numpy.fft
        threshold = 1e6

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.mean(axis=2)  # grayscale

        fft = np.abs(np.fft.fft2(img))
        high_freq = fft[fft.shape[0]//4:, fft.shape[1]//4:]
        energy = np.sum(high_freq)
        if debug:
            print(f"fft energy: {energy}")
        return energy > threshold
    
    else:
        overall_std = img.std()
        # row_var = np.var(img, axis=(1,))  # variance across each row (H, W, C)

        # Heuristic thresholds
        is_too_noisy = overall_std > 0.3
        # has_stripes = (row_var.max() - row_var.min()) > 0.05
        low_dynamic_range = overall_std < 0.02

        if debug:
            # print(f"std: {overall_std:.4f}, row_var range: {row_var.min():.4f} - {row_var.max():.4f}")
            print(f"std: {overall_std:.4f}")

        return is_too_noisy or low_dynamic_range
    