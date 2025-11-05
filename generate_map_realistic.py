import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================== PARAMETERS ====================
map_size_um = 100
resolution_um = 0.2  # µm
spectral_start = 500
spectral_end = 1800
spectral_step = 4
n_cells = 3

# Derived
x_points = int(map_size_um / resolution_um)
y_points = int(map_size_um / resolution_um)
wavenumbers = np.arange(spectral_start, spectral_end + spectral_step, spectral_step)

# Raman peaks (center, relative intensity, width)
PEAKS = {
    "protein": [(1002, 1.0, 10), (1250, 0.6, 15), (1655, 0.8, 12)],
    "lipid":   [(1300, 0.8, 20), (1445, 1.0, 15), (1745, 0.4, 10)],
    "nucleus": [(721, 0.6, 8), (785, 1.0, 10), (1090, 0.8, 12)]
}

# ==================== CELL GENERATION ====================
np.random.seed(42)
cells = []
for _ in range(n_cells):
    cx = np.random.uniform(20, 80)
    cy = np.random.uniform(20, 80)
    radius = np.random.uniform(8, 14)
    heterogeneity = {
        "protein": np.random.uniform(0.8, 1.2),
        "lipid": np.random.uniform(0.8, 1.2),
        "nucleus": np.random.uniform(0.8, 1.2)
    }
    cells.append({"center": (cx, cy), "radius": radius, "heterogeneity": heterogeneity})

# ==================== FUNCTIONS ====================
def gaussian_peak(center, amp, width):
    return amp * np.exp(-0.5 * ((wavenumbers - center) / width) ** 2)

def biochemical_spectrum(region_type, intensity=1.0):
    peaks = PEAKS[region_type]
    spec = np.zeros_like(wavenumbers, dtype=float)
    for center, amp, width in peaks:
        spec += gaussian_peak(center, amp, width)
    return intensity * spec

def fluorescence_background(level=1.0):
    coeffs = np.random.uniform([-1e-6, 0, 0], [1e-6, 1e-3, 0.1])
    poly = np.poly1d(coeffs)
    return level * (poly(wavenumbers - 500) + 0.3)

def add_cosmic_spikes(spectrum):
    n_spikes = np.random.randint(1, 5)
    for _ in range(n_spikes):
        idx = np.random.randint(0, len(spectrum))
        height = np.random.uniform(2, 8)
        width = np.random.randint(1, 3)
        spectrum[idx:idx+width] += height
    return spectrum

def radial_composition(r_norm):
    """Return biochemical composition weights based on distance from nucleus center."""
    # Define smooth transitions
    w_nucleus = np.exp(-((r_norm) / 0.35) ** 2)
    w_lipid = np.exp(-((r_norm - 0.9) / 0.25) ** 2)
    w_protein = np.exp(-((r_norm - 0.6) / 0.35) ** 2)
    # Normalize weights
    total = w_nucleus + w_lipid + w_protein
    return w_nucleus / total, w_protein / total, w_lipid / total

def generate_pixel_spectrum(x, y):
    """Generate Raman spectrum with spatially varying biochemical composition."""
    for cell in cells:
        cx, cy = cell["center"]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist < cell["radius"]:
            r_norm = dist / cell["radius"]
            w_nuc, w_prot, w_lip = radial_composition(r_norm)
            h = cell["heterogeneity"]
            spec = (
                h["nucleus"] * w_nuc * biochemical_spectrum("nucleus") +
                h["protein"] * w_prot * biochemical_spectrum("protein") +
                h["lipid"] * w_lip * biochemical_spectrum("lipid")
            )
            spec *= (0.7 + 0.3 * np.random.rand())
            # Label region
            if w_nuc > max(w_prot, w_lip):
                region = "nucleus"
            elif w_lip > max(w_nuc, w_prot):
                region = "membrane"
            else:
                region = "cytoplasm"
            break
    else:
        spec = 0.2 * biochemical_spectrum("protein") + 0.02 * np.random.randn(len(wavenumbers))
        region = "background"

    # Add background and noise
    spec += fluorescence_background(np.random.uniform(0.6, 1.2))
    spec += 0.02 * np.random.randn(len(wavenumbers))
    spec = add_cosmic_spikes(spec)
    return spec, region

# ==================== MAP GENERATION ====================
print("Generating realistic Raman map — please wait (~minutes for full 500x500 map)...")
data_rows, labels = [], []
for i, x in enumerate(np.linspace(0, map_size_um, x_points)):
    for j, y in enumerate(np.linspace(0, map_size_um, y_points)):
        spec, label = generate_pixel_spectrum(x, y)
        data_rows.append([x, y] + list(spec))
        labels.append(label)

columns = ['x (µm)', 'y (µm)'] + list(map(str, wavenumbers))
df = pd.DataFrame(data_rows, columns=columns)
df["region"] = labels

df.to_csv("synthetic_raman_map_spatial.csv", index=False)
mask = df[["x (µm)", "y (µm)", "region"]]
mask.to_csv("synthetic_raman_mask_spatial.csv", index=False)
print("✅ Saved map and mask with spatially varying composition.")

# ==================== VISUALIZATION ====================
def plot_raman_map(wavenumber_range):
    if isinstance(wavenumber_range, (int, float)):
        idx = np.argmin(np.abs(wavenumbers - wavenumber_range))
        band = df.iloc[:, idx + 2].values
        title = f"{wavenumbers[idx]:.0f} cm⁻¹"
    else:
        idx_min = np.argmin(np.abs(wavenumbers - wavenumber_range[0]))
        idx_max = np.argmin(np.abs(wavenumbers - wavenumber_range[1]))
        band = df.iloc[:, idx_min + 2:idx_max + 3].mean(axis=1).values
        title = f"{wavenumbers[idx_min]:.0f}-{wavenumbers[idx_max]:.0f} cm⁻¹"
    image = band.reshape(y_points, x_points)
    plt.figure(figsize=(6, 5))
    plt.imshow(image, cmap="inferno", origin="lower", extent=(0, map_size_um, 0, map_size_um))
    plt.colorbar(label="Intensity (a.u.)")
    plt.title(f"Raman Map – {title}")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.tight_layout()
    plt.show()

def plot_average_spectra():
    plt.figure(figsize=(7, 5))
    for region in ["nucleus", "cytoplasm", "membrane", "background"]:
        region_df = df[df["region"] == region]
        if len(region_df) > 0:
            avg_spec = region_df.iloc[:, 2:-1].mean(axis=0)
            plt.plot(wavenumbers, avg_spec, label=region)
    plt.xlabel("Raman shift (cm⁻¹)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Average Raman Spectra by Cellular Region")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot spectral maps
plot_raman_map(785)          # Nucleus (DNA)
plot_raman_map((1440, 1460)) # Membrane (lipid)
plot_raman_map(1002)         # Cytoplasm/protein
plot_average_spectra()
