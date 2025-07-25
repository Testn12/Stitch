"""
Pyramidal TIFF exporter with transformation support
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import logging
import tempfile
import shutil
import math

# Import QApplication for processEvents
from PyQt6.QtWidgets import QApplication

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

try:
    import pyvips
    PYVIPS_AVAILABLE = True
except ImportError:
    PYVIPS_AVAILABLE = False

from ..core.fragment import Fragment

class PyramidalExporter:
    """Handles export of stitched pyramidal TIFF files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Check available libraries
        if not OPENSLIDE_AVAILABLE:
            self.logger.warning("OpenSlide not available - limited pyramid support")
        if not TIFFFILE_AVAILABLE:
            self.logger.warning("tifffile not available - limited TIFF support")
        if not PYVIPS_AVAILABLE:
            self.logger.warning("pyvips not available - using fallback methods")
            
    def export_pyramidal_tiff(self, fragments: List[Fragment], output_path: str,
                             selected_levels: List[int], compression: str = "LZW",
                             tile_size: int = 256, progress_callback: Optional[Callable] = None) -> bool:
        """
        Export fragments as a stitched pyramidal TIFF
        
        Args:
            fragments: List of visible Fragment objects
            output_path: Output file path
            selected_levels: List of pyramid levels to export
            compression: Compression method ("LZW", "JPEG", "Deflate", "None")
            tile_size: Tile size for pyramid (default 256)
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            self.logger.info(f"Starting pyramidal TIFF export to {output_path}")
            
            # Filter visible fragments
            visible_fragments = [f for f in fragments if f.visible and f.file_path]
            if not visible_fragments:
                raise ValueError("No visible fragments with source files to export")
            
            # Validate selected levels
            if not selected_levels:
                raise ValueError("No pyramid levels selected for export")
                
            print(f"Exporting {len(visible_fragments)} fragments at levels {selected_levels}")
            
            # Use fallback method (more reliable for our use case)
            return self._export_with_fallback(
                visible_fragments, output_path, selected_levels,
                compression, progress_callback
            )
                
        except Exception as e:
            self.logger.error(f"Pyramidal TIFF export failed: {str(e)}")
            print(f"Export error: {e}")
            return False
            
    def _export_with_fallback(self, fragments: List[Fragment], output_path: str,
                             selected_levels: List[int], compression: str,
                             progress_callback: Optional[Callable]) -> bool:
        """Fallback export method using tifffile and numpy"""
        try:
            if not TIFFFILE_AVAILABLE:
                raise ImportError("tifffile required for fallback export")
            
            print("Using fallback export method with tifffile")
            
            # For each level, create a composite image
            level_images = []
            total_levels = len(selected_levels)
            
            for i, level in enumerate(selected_levels):
                if progress_callback:
                    progress = int((i / total_levels) * 90)
                    progress_callback(progress, f"Processing level {level}")
                
                print(f"Processing level {level} ({i+1}/{total_levels})")
                
                # Calculate bounds for this level
                bounds = self._calculate_composite_bounds_at_level(fragments, level)
                if not bounds:
                    print(f"Could not calculate bounds for level {level}")
                    continue
                    
                print(f"Level {level} bounds: {bounds}")
                    
                # Render composite for this level
                composite = self._render_composite_at_level(fragments, level, bounds)
                if composite is not None:
                    level_images.append(composite)
                    print(f"Level {level} composite shape: {composite.shape}")
                else:
                    print(f"Failed to render composite for level {level}")
            
            if not level_images:
                raise ValueError("No levels could be processed")
            
            # Save as pyramidal TIFF
            if progress_callback:
                progress_callback(95, "Saving pyramidal TIFF file...")
            
            print(f"Saving {len(level_images)} levels to {output_path}")
            
            # Configure compression
            compression_map = {
                "LZW": "lzw",
                "JPEG": "jpeg",
                "Deflate": "zlib", 
                "None": None
            }
            tiff_compression = compression_map.get(compression)
            
            # Create pyramidal structure
            # Use subifds for proper pyramid structure
            with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
                # Write base level (highest resolution)
                base_image = level_images[0]
                
                # Prepare subifds for pyramid levels
                subifds = len(level_images) - 1 if len(level_images) > 1 else 0
                
                # Write base level with subifds
                tif.write(
                    base_image,
                    compression=tiff_compression,
                    tile=(tile_size, tile_size),
                    subifds=subifds,
                    metadata={'axes': 'YXC' if len(base_image.shape) == 3 else 'YX'}
                )
                
                # Write pyramid levels as subifds
                for level_img in level_images[1:]:
                    tif.write(
                        level_img,
                        compression=tiff_compression,
                        tile=(tile_size, tile_size),
                        subfiletype=1  # Mark as reduced resolution
                    )
            
            if progress_callback:
                progress_callback(100, "Export complete")
                
            print("Pyramidal TIFF export completed successfully")
            self.logger.info("Pyramidal TIFF export completed successfully")
            return True
            
        except Exception as e:
            print(f"Fallback export failed: {e}")
            self.logger.error(f"Fallback export failed: {str(e)}")
            return False
            
    def _calculate_composite_bounds_at_level(self, fragments: List[Fragment], level: int) -> Optional[Tuple[float, float, float, float]]:
        """Calculate composite bounds at a specific pyramid level"""
        if not fragments:
            return None
            
        print(f"Calculating bounds for level {level}")
        
        # Get level 0 bounds first, then scale for the target level
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for fragment in fragments:
            # Get fragment bounds at level 0 (using current transformations)
            bbox = fragment.get_bounding_box()
            
            # Scale positions and sizes for the target level
            downsample = 2 ** level
            
            scaled_x = bbox[0] / downsample
            scaled_y = bbox[1] / downsample
            scaled_w = bbox[2] / downsample
            scaled_h = bbox[3] / downsample
            
            min_x = min(min_x, scaled_x)
            min_y = min(min_y, scaled_y)
            max_x = max(max_x, scaled_x + scaled_w)
            max_y = max(max_y, scaled_y + scaled_h)
            
        bounds = (min_x, min_y, max_x, max_y)
        print(f"Level {level} bounds: {bounds}")
        return bounds
        
    def _load_and_transform_fragment(self, fragment: Fragment, level: int) -> Optional[np.ndarray]:
        """Load fragment at specific level and apply transformations"""
        try:
            print(f"Loading fragment {fragment.name} at level {level}")
            
            # Load original image at specified level
            from ..core.image_loader import ImageLoader
            loader = ImageLoader()
            
            # Load at the specified pyramid level
            original_image = loader.load_image(fragment.file_path, level)
            if original_image is None:
                print(f"Failed to load image for fragment {fragment.name} at level {level}")
                return None
            
            print(f"Loaded image shape: {original_image.shape}")
            
            # Apply transformations (rotation, flip) to the loaded image
            # Note: Position transformations are handled during compositing
            transformed_image = self._apply_image_transforms(original_image, fragment)
            
            print(f"Transformed image shape: {transformed_image.shape}")
            return transformed_image
            
        except Exception as e:
            print(f"Failed to load fragment {fragment.name} at level {level}: {str(e)}")
            self.logger.error(f"Failed to load fragment {fragment.name} at level {level}: {str(e)}")
            return None
            
    def _apply_image_transforms(self, image: np.ndarray, fragment: Fragment) -> np.ndarray:
        """Apply rotation and flip transformations to image"""
        import cv2
        
        result = image.copy()
        
        # Apply horizontal flip
        if fragment.flip_horizontal:
            result = np.fliplr(result)
            
        # Apply vertical flip  
        if fragment.flip_vertical:
            result = np.flipud(result)
            
        # Apply rotation
        if abs(fragment.rotation) > 0.01:
            result = self._rotate_image(result, fragment.rotation)
            
        return result
        
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by arbitrary angle"""
        import cv2
        
        if abs(angle) < 0.01:
            return image
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Handle RGBA
            rotated = cv2.warpAffine(
                image, rotation_matrix, (new_width, new_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
        else:
            rotated = cv2.warpAffine(
                image, rotation_matrix, (new_width, new_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
        return rotated
            
    def _render_composite_at_level(self, fragments: List[Fragment], level: int,
                                  bounds: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Render composite image at specific pyramid level using numpy"""
        try:
            min_x, min_y, max_x, max_y = bounds
            width = int(max_x - min_x)
            height = int(max_y - min_y)
            
            if width <= 0 or height <= 0:
                print(f"Invalid dimensions for level {level}: {width}x{height}")
                return None
            
            print(f"Creating composite for level {level}: {width}x{height}")
            
            # Create composite array with alpha channel
            composite = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Render each fragment
            for fragment in fragments:
                print(f"Processing fragment {fragment.name} for level {level}")
                
                fragment_image = self._load_and_transform_fragment(fragment, level)
                if fragment_image is None:
                    print(f"Skipping fragment {fragment.name} - failed to load")
                    continue
                    
                self._composite_fragment_numpy(composite, fragment_image, fragment, bounds, level)
            
            print(f"Composite created for level {level}: {composite.shape}")
            return composite
            
        except Exception as e:
            print(f"Failed to render composite at level {level}: {e}")
            self.logger.error(f"Failed to render composite at level {level}: {str(e)}")
            return None
            
    def _composite_fragment_numpy(self, composite: np.ndarray, fragment_image: np.ndarray,
                                 fragment: Fragment, bounds: Tuple[float, float, float, float], level: int):
        """Composite fragment onto composite array using numpy"""
        try:
            min_x, min_y, _, _ = bounds
            downsample = 2 ** level
            
            # Calculate position in composite (scaled for level)
            frag_x = int((fragment.x - min_x) / downsample)
            frag_y = int((fragment.y - min_y) / downsample)
            
            print(f"Fragment {fragment.name} position at level {level}: ({frag_x}, {frag_y})")
            
            # Get dimensions
            frag_h, frag_w = fragment_image.shape[:2]
            comp_h, comp_w = composite.shape[:2]
            
            # Calculate intersection
            src_x1 = max(0, -frag_x)
            src_y1 = max(0, -frag_y)
            src_x2 = min(frag_w, comp_w - frag_x)
            src_y2 = min(frag_h, comp_h - frag_y)
            
            dst_x1 = max(0, frag_x)
            dst_y1 = max(0, frag_y)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            
            # Check overlap
            if src_x2 <= src_x1 or src_y2 <= src_y1:
                print(f"Fragment {fragment.name}: No overlap, skipping")
                return
                
            print(f"Fragment {fragment.name}: compositing region ({dst_x1},{dst_y1}) to ({dst_x2},{dst_y2})")
                
            # Extract regions
            fragment_region = fragment_image[src_y1:src_y2, src_x1:src_x2]
            
            # Ensure fragment region has alpha channel
            if len(fragment_region.shape) == 3 and fragment_region.shape[2] == 3:
                # Add alpha channel
                alpha_channel = np.full(fragment_region.shape[:2] + (1,), 255, dtype=np.uint8)
                fragment_region = np.concatenate([fragment_region, alpha_channel], axis=2)
            
            # Alpha blending
            if fragment_region.shape[2] == 4:  # RGBA
                frag_alpha = fragment_region[:, :, 3:4] / 255.0 * fragment.opacity
                frag_rgb = fragment_region[:, :, :3]
                
                comp_region = composite[dst_y1:dst_y2, dst_x1:dst_x2]
                comp_alpha = comp_region[:, :, 3:4] / 255.0
                comp_rgb = comp_region[:, :, :3]
                
                # Alpha blending
                out_alpha = frag_alpha + (1 - frag_alpha) * comp_alpha
                
                # Avoid division by zero
                mask = out_alpha[:, :, 0] > 0
                out_rgb = np.zeros_like(frag_rgb, dtype=np.float32)
                out_rgb[mask, :] = (frag_alpha[mask, :] * frag_rgb[mask, :] + 
                                   (1 - frag_alpha[mask, :]) * comp_rgb[mask, :])
                
                # Update composite
                composite[dst_y1:dst_y2, dst_x1:dst_x2, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
                composite[dst_y1:dst_y2, dst_x1:dst_x2, 3:4] = np.clip(out_alpha * 255, 0, 255).astype(np.uint8)
            else:
                # RGB fallback
                alpha = fragment.opacity
                composite[dst_y1:dst_y2, dst_x1:dst_x2, :3] = (
                    alpha * fragment_region + 
                    (1 - alpha) * composite[dst_y1:dst_y2, dst_x1:dst_x2, :3]
                ).astype(np.uint8)
                composite[dst_y1:dst_y2, dst_x1:dst_x2, 3] = 255
                
        except Exception as e:
            print(f"Failed to composite fragment {fragment.name}: {e}")
            self.logger.error(f"Failed to composite fragment: {str(e)}")