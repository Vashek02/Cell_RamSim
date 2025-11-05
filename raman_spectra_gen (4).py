import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Define characteristic Raman peaks for different organelles (wavenumber in cm^-1)
# Based on realistic biological Raman signatures with varying peak intensities and widths
ORGANELLE_SIGNATURES = {
    'nucleus': {
        # DNA/RNA signatures: DNA backbone (785), PO2- (1095), adenine (1340), guanine (1575), amide I (1660)
        'peaks': [621, 728, 785, 828, 1003, 1095, 1178, 1240, 1340, 1375, 1485, 1575, 1660],
        'intensities': [0.25, 0.35, 0.85, 0.40, 0.55, 1.2, 0.30, 0.45, 0.75, 0.35, 0.60, 0.95, 1.1],
        'widths': [8, 10, 7, 9, 8, 6, 9, 10, 7, 8, 9, 6, 8]
    },
    'mitochondria': {
        # Cytochrome C, heme proteins, respiratory chain components
        'peaks': [677, 750, 830, 920, 1003, 1125, 1208, 1340, 1450, 1585, 1655],
        'intensities': [0.40, 0.95, 0.50, 0.30, 1.1, 0.70, 0.45, 0.55, 0.85, 0.65, 0.90],
        'widths': [9, 7, 10, 11, 6, 8, 10, 9, 7, 8, 7]
    },
    'membrane': {
        # Lipids, phospholipids, cholesterol: C-C stretch (1065, 1130), CH2 def (1300, 1440), C=C (1660)
        'peaks': [702, 718, 872, 1065, 1130, 1168, 1300, 1440, 1660, 1745],
        'intensities': [0.35, 0.80, 0.40, 0.95, 0.70, 0.30, 1.0, 1.3, 0.85, 0.25],
        'widths': [10, 8, 11, 7, 8, 9, 6, 5, 7, 10]
    },
    'cytoplasm': {
        # Mixed proteins, amino acids: Phe (1003, 1032), Trp (760, 880), Pro (920), Amide bands
        'peaks': [645, 760, 830, 880, 920, 1003, 1032, 1155, 1240, 1340, 1450, 1555, 1655],
        'intensities': [0.30, 0.50, 0.65, 0.45, 0.35, 1.0, 0.85, 0.40, 0.55, 0.60, 0.95, 0.50, 1.05],
        'widths': [9, 10, 8, 11, 10, 6, 7, 10, 9, 8, 7, 9, 7]
    },
    'er': {  # Endoplasmic Reticulum
        # Protein synthesis machinery, membrane lipids
        'peaks': [665, 760, 855, 940, 1003, 1080, 1155, 1240, 1315, 1450, 1585, 1670],
        'intensities': [0.30, 0.60, 0.40, 0.35, 0.90, 0.95, 0.45, 0.70, 0.55, 1.0, 0.65, 0.95],
        'widths': [10, 9, 11, 10, 7, 7, 10, 8, 9, 6, 8, 7]
    },
    'golgi': {
        # Glycoproteins, carbohydrates, modified proteins
        'peaks': [680, 775, 835, 895, 1003, 1060, 1125, 1210, 1280, 1340, 1420, 1525, 1650],
        'intensities': [0.35, 0.75, 0.50, 0.40, 0.85, 1.05, 0.65, 0.45, 0.90, 0.55, 0.95, 0.50, 0.90],
        'widths': [9, 8, 10, 11, 7, 6, 8, 10, 7, 9, 6, 9, 7]
    },
    'lysosome': {
        # Hydrolytic enzymes, degraded material, acidic environment
        'peaks': [660, 800, 860, 935, 1003, 1030, 1110, 1185, 1260, 1325, 1380, 1465, 1640],
        'intensities': [0.30, 0.85, 0.45, 0.35, 0.80, 0.70, 0.50, 0.40, 0.95, 0.55, 0.90, 0.65, 0.95],
        'widths': [10, 8, 11, 10, 7, 8, 9, 10, 7, 9, 7, 8, 7]
    }
}

def gaussian_peak(x, center, amplitude, width):
    """Generate a Gaussian peak"""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))

def add_fluorescence_background(wavenumbers, intensity):
    """
    Add realistic fluorescence background to spectrum
    Fluorescence typically shows as a broad, sloping baseline using polynomial functions
    """
    # Normalize wavenumber to [0, 1] range for polynomial calculation
    x_norm = (wavenumbers - wavenumbers[0]) / (wavenumbers[-1] - wavenumbers[0])
    
    # Random fluorescence intensity (some cells have more than others)
    fluor_base_intensity = np.random.uniform(0.5, 3.0)
    
    # Generate random 5th order polynomial coefficients
    # These create complex, realistic fluorescence backgrounds
    a5 = np.random.uniform(-0.3, 0.3) * fluor_base_intensity
    a4 = np.random.uniform(-0.5, 0.5) * fluor_base_intensity
    a3 = np.random.uniform(-0.6, 0.6) * fluor_base_intensity
    a2 = np.random.uniform(-0.4, 0.8) * fluor_base_intensity
    a1 = np.random.uniform(-0.2, 0.5) * fluor_base_intensity
    a0 = np.random.uniform(0.8, 1.2) * fluor_base_intensity  # Constant term
    
    # Calculate 5th order polynomial background
    background = (a5 * x_norm**5 + 
                 a4 * x_norm**4 + 
                 a3 * x_norm**3 + 
                 a2 * x_norm**2 + 
                 a1 * x_norm + 
                 a0)
    
    # Ensure background is positive (fluorescence doesn't go negative)
    background = np.maximum(background, 0)
    
    # Add some slow oscillations (photobleaching effects) - optional additional complexity
    if np.random.rand() > 1.0:
        oscillation = 0.2 * fluor_base_intensity * np.sin(2 * np.pi * x_norm * np.random.uniform(1, 3))
        background += oscillation
    
    return intensity + background

def add_cosmic_spikes(wavenumbers, intensity):
    """
    Add cosmic ray spikes to spectrum
    Cosmic rays appear as narrow, very intense spikes at random positions
    """
    # Number of cosmic spikes (0-3 per spectrum, most have 0-1)
    n_spikes = np.random.choice([0, 0, 0, 1, 1, 2, 3], p=[0.4, 0.2, 0.1, 0.2, 0.08, 0.015, 0.005])
    
    if n_spikes > 0:
        # Random positions for spikes
        spike_positions = np.random.choice(len(wavenumbers), size=n_spikes, replace=False)
        
        for pos in spike_positions:
            # Cosmic spikes are very intense and very narrow
            spike_amplitude = np.random.uniform(2.0, 8.0)
            spike_width = np.random.uniform(1.5, 3.0)  # Very narrow
            
            intensity += gaussian_peak(wavenumbers, wavenumbers[pos], spike_amplitude, spike_width)
    
    return intensity

def generate_raman_spectrum(organelle, wavenumber_range=(600, 1800), num_points=1200):
    """
    Generate a synthetic Raman spectrum for a specific organelle
    
    Parameters:
    - organelle: str, name of the organelle
    - wavenumber_range: tuple, (min, max) wavenumber in cm^-1
    - num_points: int, number of data points
    
    Returns:
    - wavenumbers: array of wavenumber values
    - intensity: array of intensity values
    """
    wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], num_points)
    intensity = np.zeros(num_points)
    
    # Get organelle signature
    signature = ORGANELLE_SIGNATURES[organelle]
    
    # Add characteristic peaks with individual variations
    for peak, amp, width in zip(signature['peaks'], signature['intensities'], signature['widths']):
        # Add slight random variation to each peak (±5% intensity, ±2 cm⁻¹ position, ±15% width)
        peak_var = peak + np.random.uniform(-2, 2)
        amp_var = amp * np.random.uniform(0.95, 1.05)
        width_var = width * np.random.uniform(0.85, 1.15)
        
        intensity += gaussian_peak(wavenumbers, peak_var, amp_var, width_var)
    
    # Add minimal baseline
    baseline = 0.2 + 0.01 * np.random.rand()
    intensity += baseline
    
    # Add very broad underlying features (water, glass, general background) - much weaker
    intensity += gaussian_peak(wavenumbers, 1000, 0.08, 150)
    intensity += gaussian_peak(wavenumbers, 1600, 0.05, 120)
    
    # Reduced noise level for cleaner spectra
    noise_level = 0.15
    intensity += noise_level * np.random.randn(num_points)
    
    # Add some variation to peak positions (biological variability)
    peak_shift = np.random.randn() * 2
    intensity_roll = int(peak_shift * num_points / (wavenumber_range[1] - wavenumber_range[0]))
    intensity = np.roll(intensity, intensity_roll)
    
    # Scale intensity variation (cell-to-cell variability)
    scale_factor = 0.9 + 0.2 * np.random.rand()
    intensity *= scale_factor
    
    # Add fluorescence background (affects most biological samples)
    intensity = add_fluorescence_background(wavenumbers, intensity)
    
    # Add cosmic ray spikes (random artifacts)
    intensity = add_cosmic_spikes(wavenumbers, intensity)
    
    # Ensure no negative values
    intensity = np.maximum(intensity, 0)
    
    return wavenumbers, intensity

def generate_dataset(num_files=500, output_dir='raman_spectra'):
    """
    Generate dataset of Raman spectra CSV files
    
    Parameters:
    - num_files: int, total number of CSV files to generate
    - output_dir: str, directory to save CSV files
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    organelles = list(ORGANELLE_SIGNATURES.keys())
    
    # Distribute files across organelles
    files_per_organelle = num_files // len(organelles)
    remainder = num_files % len(organelles)
    
    file_count = 0
    
    for idx, organelle in enumerate(organelles):
        # Add remainder to first organelles
        n_files = files_per_organelle + (1 if idx < remainder else 0)
        
        for i in range(n_files):
            # Generate spectrum
            wavenumbers, intensity = generate_raman_spectrum(organelle)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Wavenumber': wavenumbers,
                'Intensity': intensity
            })
            
            # Create filename with organelle label
            filename = f"{organelle}_{i+1:03d}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            
            file_count += 1
            
            if file_count % 50 == 0:
                print(f"Generated {file_count}/{num_files} files...")
    
    print(f"\nCompleted! Generated {file_count} CSV files in '{output_dir}' directory")
    print(f"\nOrganelle distribution:")
    for organelle in organelles:
        count = len([f for f in os.listdir(output_dir) if f.startswith(organelle)])
        print(f"  {organelle}: {count} files")

def plot_average_spectra(output_dir='raman_spectra'):
    """
    Plot average Raman spectra for each organelle type
    
    Parameters:
    - output_dir: str, directory containing the CSV files
    """
    organelles = list(ORGANELLE_SIGNATURES.keys())
    
    plt.figure(figsize=(14, 10))
    
    # Store average spectra for plotting
    all_averages = {}
    
    for organelle in organelles:
        # Find all CSV files for this organelle (exclude non-CSV files)
        files = [f for f in os.listdir(output_dir) 
                if f.startswith(organelle) and f.endswith('.csv')]
        
        if not files:
            continue
        
        # Read all spectra for this organelle
        spectra = []
        wavenumbers = None
        
        for file in files:
            try:
                filepath = os.path.join(output_dir, file)
                df = pd.read_csv(filepath)
                spectra.append(df['Intensity'].values)
                if wavenumbers is None:
                    wavenumbers = df['Wavenumber'].values
            except Exception as e:
                print(f"Warning: Could not read {file}: {e}")
                continue
        
        # Calculate average spectrum
        if len(spectra) == 0:
            print(f"Warning: No valid spectra found for {organelle}")
            continue
            
        avg_spectrum = np.mean(spectra, axis=0)
        std_spectrum = np.std(spectra, axis=0)
        
        all_averages[organelle] = (wavenumbers, avg_spectrum, std_spectrum)
    
    # Plot 1: All average spectra on one plot
    plt.subplot(2, 1, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(organelles)))
    
    for idx, (organelle, color) in enumerate(zip(organelles, colors)):
        if organelle in all_averages:
            wavenumbers, avg_spectrum, _ = all_averages[organelle]
            # Offset for visualization
            offset = idx * 0.3
            plt.plot(wavenumbers, avg_spectrum + offset, label=organelle.capitalize(), 
                    color=color, linewidth=1.5)
    
    plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    plt.ylabel('Intensity (offset for clarity)', fontsize=12)
    plt.title('Average Raman Spectra by Organelle (with offset)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: All spectra overlaid without offset
    plt.subplot(2, 1, 2)
    
    for organelle, color in zip(organelles, colors):
        if organelle in all_averages:
            wavenumbers, avg_spectrum, std_spectrum = all_averages[organelle]
            plt.plot(wavenumbers, avg_spectrum, label=organelle.capitalize(), 
                    color=color, linewidth=1.5, alpha=0.7)
            # Add shaded region for standard deviation
            plt.fill_between(wavenumbers, 
                           avg_spectrum - std_spectrum, 
                           avg_spectrum + std_spectrum, 
                           color=color, alpha=0.1)
    
    plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title('Average Raman Spectra by Organelle (overlaid with std deviation)', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'average_spectra_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    for organelle in organelles:
        if organelle in all_averages:
            _, avg_spectrum, std_spectrum = all_averages[organelle]
            print(f"{organelle.capitalize():15s} - Mean Intensity: {np.mean(avg_spectrum):.3f} ± {np.mean(std_spectrum):.3f}")

if __name__ == "__main__":
    # Generate 500 CSV files
    generate_dataset(num_files=500, output_dir='raman_spectra')
    
    # Show example of first file
    print("\nExample spectrum preview (first 5 rows of first file):")
    csv_files = [f for f in os.listdir('raman_spectra') if f.endswith('.csv')]
    if csv_files:
        example_file = sorted(csv_files)[0]
        df_example = pd.read_csv(os.path.join('raman_spectra', example_file))
        print(f"\nFile: {example_file}")
        print(df_example.head())
    
    # Plot average spectra
    print("\n" + "="*60)
    print("Generating average spectra plots...")
    print("="*60)
    plot_average_spectra(output_dir='raman_spectra')