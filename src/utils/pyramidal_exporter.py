"""
Pyramidal TIFF exporter with transformation support
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import tempfile
import shutil

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
                             tile_size: int = 256, progress_callback=None) -> bool:
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
                
            # Use pyvips if available, otherwise fallback
            if PYVIPS_AVAILABLE:
                return self._export_with_pyvips(
                    visible_fragments, output_path, selected_levels, 
                    compression, tile_size, progress_callback
                )
            else:
                return self._export_with_fallback(
                    visible_fragments, output_path, selected_levels,
                    compression, progress_callback
                )
                
        except Exception as e:
            self.logger.error(f"Pyramidal TIFF export failed: {str(e)}")
            return False
            
    def _export_with_pyvips(self, fragments: List[Fragment], output_path: str,
                           selected_levels: List[int], compression: str,
                           tile_size: int, progress_callback) -> bool:
        """Export using pyvips for optimal performance"""
        try:
            # Calculate composite bounds at level 0 (full resolution)
            bounds = self._calculate_composite_bounds_at_level(fragments, 0)
            if not bounds:
                raise ValueError("Could not calculate composite bounds")
                
            min_x, min_y, max_x, max_y = bounds
            width = int(max_x - min_x)
            height = int(max_y - min_y)
            
            self.logger.info(f"Composite size at level 0: {width} x {height}")
            
            # Create base image for level 0
            base_image = None
            total_fragments = len(fragments)
            
            for i, fragment in enumerate(fragments):
                if progress_callback:
                    progress = int((i / total_fragments) * 50)  # First 50% for loading
                    progress_callback(progress, f"Processing fragment {i+1}/{total_fragments}")
                
                # Load fragment at level 0 and apply transformations
                fragment_image = self._load_and_transform_fragment(fragment, 0, bounds)
                
                if fragment_image is None:
                    continue
                    
                if base_image is None:
                    # Create base canvas
                    base_image = pyvips.Image.black(width, height, bands=4)
                    
                # Composite fragment onto base
                base_image = self._composite_fragment_pyvips(base_image, fragment_image, fragment, bounds, 0)
            
            if base_image is None:
                raise ValueError("No fragments could be processed")
            
            # Generate pyramid levels
            pyramid_images = {0: base_image}
            
            for level in selected_levels:
                if level == 0:
                    continue
                    
                if progress_callback:
                    progress = 50 + int((level / max(selected_levels)) * 30)
                    progress_callback(progress, f"Generating level {level}")
                
                # Calculate downsample factor
                downsample = 2 ** level
                level_width = max(1, width // downsample)
                level_height = max(1, height // downsample)
                
                # Resize base image for this level
                level_image = base_image.resize(level_width / width, vscale=level_height / height)
                pyramid_images[level] = level_image
            
            # Save as pyramidal TIFF
            if progress_callback:
                progress_callback(80, "Saving pyramidal TIFF...")
                
            # Configure compression
            compression_map = {
                "LZW": "lzw",
                "JPEG": "jpeg", 
                "Deflate": "deflate",
                "None": "none"
            }
            vips_compression = compression_map.get(compression, "lzw")
            
            # Save with pyramid
            save_options = {
                'compression': vips_compression,
                'tile': True,
                'tile_width': tile_size,
                'tile_height': tile_size,
                'pyramid': True,
                'bigtiff': True  # Use BigTIFF for large files
            }
            
            base_image.tiffsave(output_path, **save_options)
            
            if progress_callback:
                progress_callback(100, "Export complete")
                
            self.logger.info("Pyramidal TIFF export completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"pyvips export failed: {str(e)}")
            return False
            
    def _export_with_fallback(self, fragments: List[Fragment], output_path: str,
                             selected_levels: List[int], compression: str,
                             progress_callback) -> bool:
        """Fallback export method using tifffile and numpy"""
        try:
            if not TIFFFILE_AVAILABLE:
                raise ImportError("tifffile required for fallback export")
            
            # For fallback, we'll create separate TIFF pages for each level
            level_images = {}
            total_levels = len(selected_levels)
            
            for i, level in enumerate(selected_levels):
                if progress_callback:
                    progress = int((i / total_levels) * 90)
                    progress_callback(progress, f"Processing level {level}")
                
                # Calculate bounds for this level
                bounds = self._calculate_composite_bounds_at_level(fragments, level)
                if not bounds:
                    continue
                    
                # Render composite for this level
                composite = self._render_composite_at_level(fragments, level, bounds)
                if composite is not None:
                    level_images[level] = composite
            
            if not level_images:
                raise ValueError("No levels could be processed")
            
            # Save as multi-page TIFF
            if progress_callback:
                progress_callback(95, "Saving TIFF file...")
            
            # Configure compression
            compression_map = {
                "LZW": "lzw",
                "JPEG": "jpeg",
                "Deflate": "zlib", 
                "None": None
            }
            tiff_compression = compression_map.get(compression)
            
            # Save pages in level order
            pages = []
            for level in sorted(level_images.keys()):
                pages.append(level_images[level])
            
            tifffile.imwrite(
                output_path,
                pages,
                compression=tiff_compression,
                tile=(256, 256) if len(pages) > 1 else None,
                metadata={'axes': 'YXC'} if len(pages[0].shape) == 3 else {'axes': 'YX'}
            )
            
            if progress_callback:
                progress_callback(100, "Export complete")
                
            self.logger.info("Fallback TIFF export completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Fallback export failed: {str(e)}")
            return False
            
    def _calculate_composite_bounds_at_level(self, fragments: List[Fragment], level: int) -> Optional[Tuple[float, float, float, float]]:
        """Calculate composite bounds at a specific pyramid level"""
        if not fragments:
            return None
            
        # Get level 0 bounds first
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for fragment in fragments:
            # Get fragment bounds at level 0 (using current transformations)
            bbox = fragment.get_bounding_box()
            
            # Apply transformation scaling for the target level
            downsample = 2 ** level
            
            # Scale positions and sizes
            scaled_x = bbox[0] / downsample
            scaled_y = bbox[1] / downsample
            scaled_w = bbox[2] / downsample
            scaled_h = bbox[3] / downsample
            
            min_x = min(min_x, scaled_x)
            min_y = min(min_y, scaled_y)
            max_x = max(max_x, scaled_x + scaled_w)
            max_y = max(max_y, scaled_y + scaled_h)
            
        return (min_x, min_y, max_x, max_y)
        
    def _load_and_transform_fragment(self, fragment: Fragment, level: int, bounds: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Load fragment at specific level and apply transformations"""
        try:
            # Load original image at specified level
            from ..core.image_loader import ImageLoader
            loader = ImageLoader()
            
            # Load at the specified pyramid level
            original_image = loader.load_image(fragment.file_path, level)
            if original_image is None:
                return None
            
            # Apply transformations (rotation, flip) to the loaded image
            # Note: Position transformations are handled during compositing
            transformed_image = self._apply_image_transforms(original_image, fragment, level)
            
            return transformed_image
            
        except Exception as e:
            self.logger.error(f"Failed to load fragment {fragment.name} at level {level}: {str(e)}")
            return None
            
    def _apply_image_transforms(self, image: np.ndarray, fragment: Fragment, level: int) -> np.ndarray:
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
        
    def _composite_fragment_pyvips(self, base_image, fragment_image, fragment: Fragment, 
                                  bounds: Tuple[float, float, float, float], level: int):
        """Composite fragment onto base image using pyvips"""
        try:
            # Calculate position in composite (scaled for level)
            min_x, min_y, _, _ = bounds
            downsample = 2 ** level
            
            # Scale fragment position for this level
            frag_x = int((fragment.x - min_x) / downsample)
            frag_y = int((fragment.y - min_y) / downsample)
            
            # Convert numpy array to pyvips image
            if isinstance(fragment_image, np.ndarray):
                height, width = fragment_image.shape[:2]
                bands = fragment_image.shape[2] if len(fragment_image.shape) == 3 else 1
                
                vips_image = pyvips.Image.new_from_memory(
                    fragment_image.tobytes(),
                    width, height, bands,
                    'uchar'
                )
            else:
                vips_image = fragment_image
            
            # Composite with alpha blending
            result = base_image.composite(vips_image, 'over', x=frag_x, y=frag_y)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to composite fragment: {str(e)}")
            return base_image
            
    def _render_composite_at_level(self, fragments: List[Fragment], level: int,
                                  bounds: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Render composite image at specific pyramid level using numpy"""
        try:
            min_x, min_y, max_x, max_y = bounds
            width = int(max_x - min_x)
            height = int(max_y - min_y)
            
            if width <= 0 or height <= 0:
                return None
            
            # Create composite array
            composite = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Render each fragment
            for fragment in fragments:
                fragment_image = self._load_and_transform_fragment(fragment, level, bounds)
                if fragment_image is None:
                    continue
                    
                self._composite_fragment_numpy(composite, fragment_image, fragment, bounds, level)
            
            return composite
            
        except Exception as e:
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
                return
                
            # Extract regions
            fragment_region = fragment_image[src_y1:src_y2, src_x1:src_x2]
            
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
            self.logger.error(f"Failed to composite fragment: {str(e)}")