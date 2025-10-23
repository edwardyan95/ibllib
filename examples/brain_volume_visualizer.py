"""
Interactive 3D Brain Volume Visualizer for Allen Institute Mouse Brain Atlas

This script provides an interactive interface to visualize and slice through
the Allen Institute 3D mouse brain volume with rotation controls.

Usage:
    python brain_volume_visualizer.py [path_to_template_volume.npy]
    
If no path is provided, it will look for 'template_volume_10um.npy' in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import argparse
from pathlib import Path

class BrainVolumeVisualizer:
    def __init__(self, volume_path, crop_region=None, downsample_factor=1):
        """Initialize the brain volume visualizer."""
        self.full_volume = np.load(volume_path)
        print(f"Loaded brain volume with shape: {self.full_volume.shape}")
        
        # Apply cropping if specified
        if crop_region is not None:
            z_start, z_end, y_start, y_end = crop_region
            self.volume = self.full_volume[z_start:z_end, y_start:y_end, :]
            print(f"Cropped volume shape: {self.volume.shape}")
        else:
            self.volume = self.full_volume
        
        # Apply downsampling for speed if specified
        if downsample_factor > 1:
            self.volume = self.volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
            print(f"Downsampled volume shape: {self.volume.shape} (factor: {downsample_factor})")
        
        # Initialize rotation angles (in degrees)
        self.pitch = 0  # rotation around x-axis (nose up/down)
        self.roll = 0   # rotation around z-axis (roll left/right)
        self.yaw = 0    # rotation around y-axis (turn left/right)
        
        # Slice parameters
        self.slice_height = self.volume.shape[0] // 2  # Default to middle slice
        self.slice_axis = 0  # 0=coronal, 1=sagittal, 2=horizontal
        
        # Create the figure and axes
        self.setup_plot()
        
    def setup_plot(self):
        """Set up the matplotlib figure and widgets."""
        self.fig = plt.figure(figsize=(15, 10))
        
        # Main image axis
        self.ax_image = plt.subplot2grid((3, 5), (0, 0), colspan=4, rowspan=2)
        self.ax_image.set_title('Brain Volume Slice')
        
        # Control axes
        self.ax_pitch = plt.subplot2grid((3, 5), (0, 4))
        self.ax_roll = plt.subplot2grid((3, 5), (1, 4))
        self.ax_yaw = plt.subplot2grid((3, 5), (2, 0))
        self.ax_slice = plt.subplot2grid((3, 5), (2, 1))
        self.ax_axis = plt.subplot2grid((3, 5), (2, 2))
        self.ax_reset = plt.subplot2grid((3, 5), (2, 3))
        
        # Create sliders with more precise control
        self.slider_pitch = Slider(self.ax_pitch, 'Pitch', -90, 90, valinit=self.pitch, valfmt='%.2f°', valstep=0.1)
        self.slider_roll = Slider(self.ax_roll, 'Roll', -180, 180, valinit=self.roll, valfmt='%.2f°', valstep=0.1)
        self.slider_yaw = Slider(self.ax_yaw, 'Yaw', -180, 180, valinit=self.yaw, valfmt='%.2f°', valstep=0.1)
        
        # Slice height slider with more precise control
        max_slice = self.volume.shape[self.slice_axis] - 1
        self.slider_slice = Slider(self.ax_slice, 'Slice', 0, max_slice, 
                                 valinit=self.slice_height, valfmt='%.1f', valstep=0.1)
        
        # Axis selection radio buttons
        self.radio_axis = RadioButtons(self.ax_axis, ('Coronal (A-P)', 'Sagittal (L-R)', 'Horizontal (D-V)'))
        
        # Control buttons
        self.btn_reset = Button(self.ax_reset, 'Reset')
        
        # Add export button
        self.ax_export = plt.subplot2grid((3, 5), (2, 4))
        self.btn_export = Button(self.ax_export, 'Export Slice')
        
        # Connect callbacks
        self.slider_pitch.on_changed(self.update_pitch)
        self.slider_roll.on_changed(self.update_roll)
        self.slider_yaw.on_changed(self.update_yaw)
        self.slider_slice.on_changed(self.update_slice)
        self.radio_axis.on_clicked(self.update_axis)
        self.btn_reset.on_clicked(self.reset_view)
        self.btn_export.on_clicked(self.export_slice)
        
        # Initial plot
        self.update_display()
        
    def get_rotated_slice_fast(self, volume, axis, slice_idx, pitch, roll, yaw):
        """Get a rotated slice with optimized performance."""
        # Convert degrees to radians - FIXED: swap pitch and yaw parameters
        pitch_rad = np.radians(yaw)  # Use yaw value for pitch rotation
        roll_rad = np.radians(roll)
        yaw_rad = np.radians(pitch)  # Use pitch value for yaw rotation
        
        # Pre-compute trigonometric values
        cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
        cr, sr = np.cos(roll_rad), np.sin(roll_rad)
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        
        # Create rotation matrix (optimized) - FIXED: swapped yaw and pitch
        R = np.array([
            [cp*cy, -cp*sy, sp],
            [sy*cr + cy*sp*sr, cy*cr - sy*sp*sr, -cp*sr],
            [sy*sr - cy*sp*cr, cy*sr + sy*sp*cr, cp*cr]
        ])
        
        # Get volume center
        center = np.array(volume.shape) / 2
        
        # Create 2D coordinate grid for the slice (vectorized)
        # Allen atlas convention: axis 0=anterior-posterior, axis 1=lateral, axis 2=dorsal-ventral
        if axis == 0:  # Coronal slice (anterior-posterior)
            # Fixed z (anterior-posterior), vary y (lateral) and x (dorsal-ventral)
            y, x = np.meshgrid(
                np.arange(volume.shape[1]) - center[1],  # lateral (left-right)
                np.arange(volume.shape[2]) - center[2],  # dorsal-ventral (up-down)
                indexing='ij'
            )
            z = np.full_like(y, slice_idx - center[0])  # anterior-posterior (front-back)
        elif axis == 1:  # Sagittal slice (lateral) - FIXED: now shows horizontal view
            # Fixed x (dorsal-ventral), vary z (anterior-posterior) and y (lateral)
            z, y = np.meshgrid(
                np.arange(volume.shape[0]) - center[0],  # anterior-posterior (front-back)
                np.arange(volume.shape[1]) - center[1],  # lateral (left-right)
                indexing='ij'
            )
            x = np.full_like(z, slice_idx - center[2])  # dorsal-ventral (up-down)
        else:  # Horizontal slice (dorsal-ventral) - FIXED: now shows sagittal view
            # Fixed y (lateral), vary z (anterior-posterior) and x (dorsal-ventral)
            z, x = np.meshgrid(
                np.arange(volume.shape[0]) - center[0],  # anterior-posterior (front-back)
                np.arange(volume.shape[2]) - center[2],  # dorsal-ventral (up-down)
                indexing='ij'
            )
            y = np.full_like(z, slice_idx - center[1])  # lateral (left-right)
        
        # Stack coordinates and apply rotation (vectorized)
        coords = np.stack([z.flatten(), y.flatten(), x.flatten()], axis=1)
        rotated_coords = coords @ R.T
        
        # Add center back
        rotated_coords += center
        
        # Interpolate only the slice (use lower order for speed)
        from scipy.ndimage import map_coordinates
        rotated_slice = map_coordinates(volume, rotated_coords.T, 
                                      order=0, mode='constant', cval=0)  # order=0 for speed
        
        return rotated_slice.reshape(z.shape)
    
    def get_slice(self, volume, axis, slice_idx):
        """Extract a slice from the volume along the specified axis."""
        # Allen atlas convention: axis 0=anterior-posterior, axis 1=lateral, axis 2=dorsal-ventral
        if axis == 0:  # Coronal slice (anterior-posterior)
            return volume[slice_idx, :, :]  # Fixed anterior-posterior, show lateral x dorsal-ventral
        elif axis == 1:  # Sagittal slice (lateral) - FIXED: swap with horizontal
            return volume[:, :, slice_idx]  # Fixed dorsal-ventral, show anterior-posterior x lateral
        elif axis == 2:  # Horizontal slice (dorsal-ventral) - FIXED: swap with sagittal
            return volume[:, slice_idx, :]  # Fixed lateral, show anterior-posterior x dorsal-ventral
    
    def update_display(self):
        """Update the displayed slice."""
        # Get rotated slice directly (much faster)
        slice_data = self.get_rotated_slice_fast(self.volume, self.slice_axis, self.slice_height, 
                                               self.pitch, self.roll, self.yaw)
        
        # Clear and update the image
        self.ax_image.clear()
        # Fix orientation: flip vertically to show brain right-side up
        self.ax_image.imshow(np.flipud(slice_data), cmap='gray', origin='lower')
        
        # Update title with current parameters
        axis_names = ['Coronal (A-P)', 'Sagittal (L-R)', 'Horizontal (D-V)']
        self.ax_image.set_title(f'{axis_names[self.slice_axis]} Slice {self.slice_height:.1f} '
                               f'(Pitch: {self.pitch:.2f}°, Roll: {self.roll:.2f}°, Yaw: {self.yaw:.2f}°)')
        
        # Set appropriate axis labels based on slice type
        if self.slice_axis == 0:  # Coronal
            self.ax_image.set_xlabel('Dorsal-Ventral')
            self.ax_image.set_ylabel('Left-Right')
        elif self.slice_axis == 1:  # Sagittal
            self.ax_image.set_xlabel('Dorsal-Ventral')
            self.ax_image.set_ylabel('Anterior-Posterior')
        else:  # Horizontal
            self.ax_image.set_xlabel('Left-Right')
            self.ax_image.set_ylabel('Anterior-Posterior')
        
        # Update slice slider range (use original volume dimensions)
        max_slice = self.volume.shape[self.slice_axis] - 1
        self.slider_slice.valmax = max_slice
        self.slider_slice.ax.set_xlim(0, max_slice)
        
        self.fig.canvas.draw()
    
    def update_pitch(self, val):
        """Update pitch angle."""
        self.pitch = val
        self.update_display()
    
    def update_roll(self, val):
        """Update roll angle."""
        self.roll = val
        self.update_display()
    
    def update_yaw(self, val):
        """Update yaw angle."""
        self.yaw = val
        self.update_display()
    
    def update_slice(self, val):
        """Update slice index."""
        self.slice_height = int(val)
        self.update_display()
    
    def update_axis(self, label):
        """Update slice axis."""
        axis_map = {'Coronal (A-P)': 0, 'Sagittal (L-R)': 1, 'Horizontal (D-V)': 2}
        self.slice_axis = axis_map[label]
        self.slice_height = self.volume.shape[self.slice_axis] // 2
        self.slider_slice.set_val(self.slice_height)
        self.update_display()
    
    def reset_view(self, event):
        """Reset all parameters to default values."""
        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.slice_height = self.volume.shape[self.slice_axis] // 2
        
        self.slider_pitch.reset()
        self.slider_roll.reset()
        self.slider_yaw.reset()
        self.slider_slice.reset()
        
        self.update_display()
    
    def get_current_slice(self):
        """Get the current 2D atlas matrix for the visualized slice."""
        slice_data = self.get_rotated_slice_fast(self.volume, self.slice_axis, self.slice_height, 
                                               self.pitch, self.roll, self.yaw)
        # Flip to match display orientation
        return np.flipud(slice_data)
    
    def export_slice(self, event):
        """Export the current slice as a numpy array and save parameters."""
        import os
        from datetime import datetime
        
        # Get current slice
        current_slice = self.get_current_slice()
        
        # Create filename with current parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        axis_names = ['coronal', 'sagittal', 'horizontal']
        filename = f"atlas_slice_{axis_names[self.slice_axis]}_{timestamp}.npy"
        
        # Save the slice
        np.save(filename, current_slice)
        
        # Save parameters
        params_filename = f"atlas_params_{axis_names[self.slice_axis]}_{timestamp}.txt"
        with open(params_filename, 'w') as f:
            f.write(f"Atlas Slice Parameters\n")
            f.write(f"=====================\n")
            f.write(f"Slice Type: {axis_names[self.slice_axis]}\n")
            f.write(f"Slice Index: {self.slice_height:.1f}\n")
            f.write(f"Pitch: {self.pitch:.2f}°\n")
            f.write(f"Roll: {self.roll:.2f}°\n")
            f.write(f"Yaw: {self.yaw:.2f}°\n")
            f.write(f"Slice Shape: {current_slice.shape}\n")
            f.write(f"Volume Shape: {self.volume.shape}\n")
            f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Exported slice: {filename}")
        print(f"Exported parameters: {params_filename}")
        print(f"Slice shape: {current_slice.shape}")
        print(f"Parameters: Pitch={self.pitch:.2f}°, Roll={self.roll:.2f}°, Yaw={self.yaw:.2f}°")
        
        return current_slice
    
    def get_slice_matrix(self):
        """Get the current 2D atlas matrix without exporting files."""
        return self.get_current_slice()
    
    def get_slice_info(self):
        """Get information about the current slice."""
        axis_names = ['coronal', 'sagittal', 'horizontal']
        return {
            'slice_type': axis_names[self.slice_axis],
            'slice_index': self.slice_height,
            'pitch': self.pitch,
            'roll': self.roll,
            'yaw': self.yaw,
            'slice_shape': self.get_current_slice().shape,
            'volume_shape': self.volume.shape
        }
    
    def show(self):
        """Display the interactive visualizer."""
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Interactive 3D Brain Volume Visualizer')
    parser.add_argument('volume_path', nargs='?', default='template_volume_10um.npy',
                       help='Path to the brain volume .npy file')
    parser.add_argument('--crop', action='store_true', 
                       help='Apply the standard crop region [999:1230, 199:940]')
    parser.add_argument('--crop-custom', nargs=4, type=int, metavar=('Z_START', 'Z_END', 'Y_START', 'Y_END'),
                       help='Custom crop region: z_start z_end y_start y_end')
    parser.add_argument('--downsample', type=int, default=1, metavar='FACTOR',
                       help='Downsample factor for speed (2=half resolution, 4=quarter resolution)')
    args = parser.parse_args()
    
    volume_path = Path(args.volume_path)
    if not volume_path.exists():
        print(f"Error: Volume file not found: {volume_path}")
        print("Please provide the correct path to your template_volume_10um.npy file")
        return
    
    # Determine crop region
    crop_region = None
    if args.crop:
        crop_region = (999, 1230, 199, 940)  # Standard crop
        print("Using standard crop region: [999:1230, 199:940]")
    elif args.crop_custom:
        crop_region = tuple(args.crop_custom)
        print(f"Using custom crop region: [{crop_region[0]}:{crop_region[1]}, {crop_region[2]}:{crop_region[3]}]")
    
    # Create and show the visualizer
    visualizer = BrainVolumeVisualizer(volume_path, crop_region, args.downsample)
    visualizer.show()

if __name__ == "__main__":
    main()
