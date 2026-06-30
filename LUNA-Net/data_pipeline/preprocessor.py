import numpy as np
from typing import List, Tuple

class LunarDataPreprocessor:
    """
    Handles preprocessing for Chandrayaan-2 OHRC (Optical) and DFSAR (Radar) data.
    Includes min-max normalization, geometric alignment, and patch generation.
    """
    def __init__(self, patch_size: int = 256, stride: int = 256):
        """
        Initializes the preprocessor.

        Args:
            patch_size (int): Size of the square patches to generate (e.g., 256).
            stride (int): Stride for the sliding window. Defaults to patch_size (no overlap).
        """
        self.patch_size = patch_size
        self.stride = stride

    def _min_max_normalize(self, array: np.ndarray) -> np.ndarray:
        """
        Applies min-max normalization to scale data between 0 and 1.
        
        Args:
            array (np.ndarray): The input data array.
            
        Returns:
            np.ndarray: The normalized array.
        """
        min_val = np.min(array)
        max_val = np.max(array)
        if max_val - min_val == 0:
            return np.zeros_like(array)
        return (array - min_val) / (max_val - min_val)

    def _radiometric_calibration_stub(self, array: np.ndarray, is_radar: bool = False) -> np.ndarray:
        """
        Applies radiometric calibration to convert Digital Numbers (DN) to physical units.
        This is a stub demonstrating where instrument-specific logic goes.
        
        Args:
            array (np.ndarray): The uncalibrated data array.
            is_radar (bool): Flag indicating if the data is radar (DFSAR).
            
        Returns:
            np.ndarray: The calibrated array.
        """
        if is_radar:
            # Example: Convert to Sigma-Nought (Backscatter coefficient)
            # mock calibration factor
            calibrated = array * 0.85 + 0.1
        else:
            # Example: Convert to Top-of-Atmosphere (TOA) Reflectance
            calibrated = array * 0.9 + 0.05
        return calibrated

    def geometric_alignment(self, optical_array: np.ndarray, radar_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aligns the optical and radar arrays spatially to ensure they cover the same physical area.
        
        Args:
            optical_array (np.ndarray): The optical data array.
            radar_array (np.ndarray): The radar data array.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: The geometrically aligned optical and radar arrays.
        """
        # For the hackathon context, we assume they are roughly aligned or 
        # we crop them to the minimum common dimensions.
        min_h = min(optical_array.shape[0], radar_array.shape[0])
        min_w = min(optical_array.shape[1], radar_array.shape[1])
        
        aligned_opt = optical_array[:min_h, :min_w]
        aligned_rad = radar_array[:min_h, :min_w]
        return aligned_opt, aligned_rad

    def process_and_patchify(self, optical_array: np.ndarray, radar_array: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Runs the full preprocessing pipeline: Calibration -> Alignment -> Normalization -> Patching.
        
        Args:
            optical_array (np.ndarray): Raw optical data array.
            radar_array (np.ndarray): Raw radar data array.
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: A list of tuples containing (optical_patch, radar_patch).
        """
        # 1. Radiometric Calibration
        opt_calib = self._radiometric_calibration_stub(optical_array, is_radar=False)
        rad_calib = self._radiometric_calibration_stub(radar_array, is_radar=True)

        # 2. Geometric Alignment
        opt_aligned, rad_aligned = self.geometric_alignment(opt_calib, rad_calib)

        # 3. Normalization
        opt_norm = self._min_max_normalize(opt_aligned)
        rad_norm = self._min_max_normalize(rad_aligned)

        # Handle channels. If 2D (H, W), add a channel dimension (H, W, 1).
        if len(opt_norm.shape) == 2:
            opt_norm = np.expand_dims(opt_norm, axis=-1)
        if len(rad_norm.shape) == 2:
            rad_norm = np.expand_dims(rad_norm, axis=-1)

        # Ensure multi-channel configuration (e.g., repeating 1 channel to 3 if required by ViT later, 
        # though standard practice is to handle it in the model definition. We'll keep them as is).
        
        # 4. Patch Generation (Sliding Window)
        h, w = opt_norm.shape[:2]
        patches = []
        
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                opt_patch = opt_norm[i : i + self.patch_size, j : j + self.patch_size, :]
                rad_patch = rad_norm[i : i + self.patch_size, j : j + self.patch_size, :]
                patches.append((opt_patch, rad_patch))
                
        return patches
