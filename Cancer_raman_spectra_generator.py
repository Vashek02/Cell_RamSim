import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_spectra = 750  # Total number of spectra
n_healthy = 375  # Half healthy
n_cancer = 375   # Half cancer
wavenumber_range = (400, 1800)  # Typical fingerprint region for biological samples
n_points = 1400  # Number of data points

# Generate wavenumber axis
wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], n_points)

# Define characteristic Raman bands for biological/plasma samples
# Format: (position, intensity, width)
common_bands = [
    (621, 0.25, 15),   # C-C twist (Phenylalanine)
    (643, 0.18, 12),   # C-C twist (Tyrosine)
    (758, 0.22, 18),   # Tryptophan
    (828, 0.15, 14),   # Tyrosine, proline
    (853, 0.20, 16),   # Tyrosine, proline, hydroxyproline
    (875, 0.17, 13),   # Tryptophan
    (937, 0.19, 15),   # Proline, valine, protein backbone
    (1003, 0.35, 18),  # Phenylalanine (strong band)
    (1032, 0.20, 14),  # Proline, phenylalanine
    (1065, 0.16, 12),  # C-C lipid
    (1126, 0.18, 15),  # C-C proteins
    (1157, 0.22, 17),  # Carotenoids, proteins
    (1207, 0.19, 14),  # Tryptophan, phenylalanine
    (1245, 0.24, 16),  # Amide III
    (1301, 0.21, 15),  # CH2 twist lipids
    (1337, 0.18, 13),  # Tryptophan, nucleic acids
    (1449, 0.28, 19),  # CH2 deformation
    (1583, 0.17, 14),  # Nucleic acids
    (1606, 0.15, 12),  # C=C phenylalanine, tyrosine
    (1656, 0.30, 20),  # Amide I, C=C lipids
]

# Subtle differences for cancer vs healthy (very small changes to make classification challenging)
cancer_band_modifiers = {
    1003: 1.08,  # Slight increase in phenylalanine
    1245: 0.94,  # Slight decrease in Amide III
    1337: 1.12,  # Increase in nucleic acids (proliferation)
    1449: 1.06,  # Slight increase in lipids
    1656: 0.96,  # Slight decrease in Amide I
    828: 0.93,   # Decrease in certain amino acids
    1583: 1.15,  # Increase in nucleic acids
}

def generate_baseline(wavenumbers, complexity='medium'):
    """Generate fluorescence baseline"""
    x = (wavenumbers - wavenumbers[0]) / (wavenumbers[-1] - wavenumbers[0])
    
    # Polynomial baseline (fluorescence-like)
    baseline = 2.5 + 1.8*x - 1.2*x**2 + 0.6*x**3
    
    # Add smooth variations
    n_variations = np.random.randint(2, 5)
    for _ in range(n_variations):
        center = np.random.uniform(0.2, 0.8)
        width = np.random.uniform(0.15, 0.4)
        amplitude = np.random.uniform(0.3, 0.8)
        baseline += amplitude * np.exp(-((x - center)**2) / (2 * width**2))
    
    return baseline

def add_cosmic_ray_spikes(spectrum, wavenumbers, n_spikes=None):
    """Add cosmic ray spikes (sharp, intense artifacts)"""
    if n_spikes is None:
        n_spikes = np.random.poisson(0.3)  # Average 2 spikes per spectrum
    
    spectrum_copy = spectrum.copy()
    for _ in range(n_spikes):
        spike_pos = np.random.randint(50, len(spectrum) - 50)
        spike_intensity = np.random.uniform(1.5, 4.0) * np.max(spectrum)
        spike_width = np.random.randint(1, 3)
        
        for i in range(max(0, spike_pos - spike_width), min(len(spectrum), spike_pos + spike_width + 1)):
            distance = abs(i - spike_pos)
            spectrum_copy[i] += spike_intensity * (1 - distance / (spike_width + 1))
    
    return spectrum_copy

def generate_raman_spectrum(wavenumbers, bands, is_cancer=False, biological_var=0.15):
    """Generate a single realistic Raman spectrum"""
    
    # Initialize spectrum
    spectrum = np.zeros_like(wavenumbers)
    
    # Add Raman bands
    for pos, intensity, width in bands:
        # Apply cancer modifiers
        if is_cancer and pos in cancer_band_modifiers:
            intensity *= cancer_band_modifiers[pos]
        
        # Biological variation
        intensity *= np.random.normal(1.0, biological_var)
        width *= np.random.normal(1.0, 0.1)
        pos += np.random.normal(0, 1.5)  # Slight band position variation
        
        # Add Gaussian band
        spectrum += intensity * np.exp(-((wavenumbers - pos)**2) / (2 * width**2))
    
    # Add fluorescence baseline
    baseline = generate_baseline(wavenumbers)
    baseline *= np.random.uniform(0.7, 1.3)  # Variation in baseline intensity
    spectrum += baseline
    
    # Add noise (shot noise, detector noise)
    noise_level = np.random.uniform(0.03, 0.08)
    noise = np.random.normal(0, noise_level, len(wavenumbers))
    spectrum += noise * np.mean(spectrum)
    
    # Add 1/f noise (low frequency drift)
    freqs = np.fft.rfftfreq(len(wavenumbers))
    freqs[0] = 1  # Avoid division by zero
    pink_noise = np.fft.irfft(np.fft.rfft(np.random.randn(len(wavenumbers))) / np.sqrt(freqs))
    pink_noise = pink_noise[:len(wavenumbers)]
    spectrum += 0.02 * pink_noise * np.mean(spectrum)
    
    # Add cosmic ray spikes (randomly)
    if np.random.random() < 0.25:  # 25% chance of spikes
        spectrum = add_cosmic_ray_spikes(spectrum, wavenumbers)
    
    # Ensure non-negative values
    spectrum = np.maximum(spectrum, 0)
    
    # Add subtle instrumental variations
    spectrum *= np.random.normal(1.0, 0.05)
    
    return spectrum

# Generate all spectra
print("Generating Raman spectra...")
print(f"Healthy samples: {n_healthy}")
print(f"Cancer samples: {n_cancer}")

all_spectra = []
labels = []

# Generate healthy spectra
for i in range(n_healthy):
    spectrum = generate_raman_spectrum(wavenumbers, common_bands, is_cancer=False, biological_var=0.15)
    all_spectra.append(spectrum)
    labels.append('Healthy')
    if (i + 1) % 100 == 0:
        print(f"  Generated {i + 1}/{n_healthy} healthy spectra")

# Generate cancer spectra
for i in range(n_cancer):
    spectrum = generate_raman_spectrum(wavenumbers, common_bands, is_cancer=True, biological_var=0.15)
    all_spectra.append(spectrum)
    labels.append('Cancer')
    if (i + 1) % 100 == 0:
        print(f"  Generated {i + 1}/{n_cancer} cancer spectra")

# Create DataFrame
print("\nCreating DataFrame...")
df = pd.DataFrame(all_spectra, columns=[f'{wn:.2f}' for wn in wavenumbers])
df.insert(0, 'Label', labels)
df.insert(0, 'Sample_ID', [f'Sample_{i:04d}' for i in range(len(labels))])

# Save to CSV
output_file = 'raman_spectra_cancer_vs_healthy.csv'
df.to_csv(output_file, index=False)
print(f"\nSpectra saved to: {output_file}")
print(f"Total spectra: {len(df)}")
print(f"Shape: {df.shape}")

# Plot average spectra with standard deviations
print("\nGenerating plots...")

healthy_spectra = np.array([all_spectra[i] for i in range(len(labels)) if labels[i] == 'Healthy'])
cancer_spectra = np.array([all_spectra[i] for i in range(len(labels)) if labels[i] == 'Cancer'])

healthy_mean = np.mean(healthy_spectra, axis=0)
healthy_std = np.std(healthy_spectra, axis=0)
cancer_mean = np.mean(cancer_spectra, axis=0)
cancer_std = np.std(cancer_spectra, axis=0)

# Create figure with multiple subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Average spectra with std
ax1 = axes[0]
ax1.plot(wavenumbers, healthy_mean, 'b-', linewidth=2, label='Healthy (mean)', alpha=0.8)
ax1.fill_between(wavenumbers, healthy_mean - healthy_std, healthy_mean + healthy_std, 
                  color='blue', alpha=0.2, label='Healthy (±1 SD)')
ax1.plot(wavenumbers, cancer_mean, 'r-', linewidth=2, label='Cancer (mean)', alpha=0.8)
ax1.fill_between(wavenumbers, cancer_mean - cancer_std, cancer_mean + cancer_std, 
                  color='red', alpha=0.2, label='Cancer (±1 SD)')
ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
ax1.set_title('Average Raman Spectra: Cancer vs Healthy', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Difference spectrum (Cancer - Healthy)
ax2 = axes[1]
difference = cancer_mean - healthy_mean
ax2.plot(wavenumbers, difference, 'g-', linewidth=2, label='Difference (Cancer - Healthy)')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.fill_between(wavenumbers, 0, difference, where=(difference > 0), color='red', alpha=0.3, label='Cancer higher')
ax2.fill_between(wavenumbers, 0, difference, where=(difference < 0), color='blue', alpha=0.3, label='Healthy higher')
ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
ax2.set_ylabel('Intensity Difference (a.u.)', fontsize=12)
ax2.set_title('Difference Spectrum (Subtle variations for ML challenge)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Sample individual spectra
ax3 = axes[2]
n_samples = 5
for i in range(n_samples):
    ax3.plot(wavenumbers, healthy_spectra[i], 'b-', alpha=0.4, linewidth=1)
    ax3.plot(wavenumbers, cancer_spectra[i], 'r-', alpha=0.4, linewidth=1)

# Add legend with proxy artists
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='b', lw=2, alpha=0.6, label=f'Healthy (n={n_samples})'),
                   Line2D([0], [0], color='r', lw=2, alpha=0.6, label=f'Cancer (n={n_samples})')]
ax3.legend(handles=legend_elements, loc='upper right', fontsize=10)
ax3.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
ax3.set_ylabel('Intensity (a.u.)', fontsize=12)
ax3.set_title('Sample Individual Spectra (showing biological variation)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('raman_spectra_comparison.png', dpi=300, bbox_inches='tight')
print("Plots saved to: raman_spectra_comparison.png")

# Print statistics
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)
print(f"Total spectra: {len(df)}")
print(f"Healthy: {n_healthy} ({100*n_healthy/len(df):.1f}%)")
print(f"Cancer: {n_cancer} ({100*n_cancer/len(df):.1f}%)")
print(f"Wavenumber range: {wavenumber_range[0]} - {wavenumber_range[1]} cm⁻¹")
print(f"Spectral resolution: {n_points} points")
print(f"\nMean intensity - Healthy: {healthy_mean.mean():.3f}")
print(f"Mean intensity - Cancer: {cancer_mean.mean():.3f}")
print(f"Relative difference: {100*abs(cancer_mean.mean() - healthy_mean.mean())/healthy_mean.mean():.2f}%")
print("\nKey cancer-related band modifications:")
for band_pos, modifier in cancer_band_modifiers.items():
    print(f"  {band_pos} cm⁻¹: {100*(modifier-1):+.1f}% change")
print("="*60)

plt.show()
