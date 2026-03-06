import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --- 1. SET UP DATA (Feb 23, 2026 Readings) ---
def normalize_gps(lat, lon):
    return (lat - 28.4) / 0.4, (lon - 77.0) / 0.4

def denormalize_gps(lat_n, lon_n):
    return lat_n * 0.4 + 28.4, lon_n * 0.4 + 77.0

# Current Sensor Readings (Delhi/Ghaziabad)
sensors = [
    {"name": "Anand Vihar", "coords": normalize_gps(28.65, 77.30), "pm25": 460.0},
    {"name": "Loni (Hotspot)", "coords": normalize_gps(28.69, 77.39), "pm25": 510.0},
    {"name": "ITO",         "coords": normalize_gps(28.62, 77.24), "pm25": 195.0},
    {"name": "RK Puram",    "coords": normalize_gps(28.56, 77.19), "pm25": 169.0},
    {"name": "Sonia Vihar", "coords": normalize_gps(28.71, 77.25), "pm25": 169.0}
]

sensor_coords = torch.tensor([s["coords"] for s in sensors], dtype=torch.float32)
sensor_values = torch.tensor([[s["pm25"]] for s in sensors], dtype=torch.float32)

# --- 2. NEURAL FIELD MODEL ---
class DelhiField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)

model = DelhiField()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# --- 3. TRAINING ---
grid_res = 50
x_range = torch.linspace(0, 1, grid_res)
y_range = torch.linspace(0, 1, grid_res)
grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
city_grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

for epoch in range(1001):
    optimizer.zero_grad()
    preds = model(sensor_coords)
    data_loss = torch.mean((preds - sensor_values)**2)

    # Physics Loss (Smooth Diffusion)
    city_grid.requires_grad = True
    c = model(city_grid)
    grads = torch.autograd.grad(c.sum(), city_grid, create_graph=True)[0]
    physics_loss = torch.mean(grads**2)

    (data_loss + 0.1 * physics_loss).backward()
    optimizer.step()

# --- 4. OUTPUT & DETECTIVE REPORT ---
with torch.no_grad():
    field = model(city_grid).reshape(grid_res, grid_res).numpy()

    # Find the peak coordinate
    idx = np.unravel_index(np.argmax(field), field.shape)
    origin_lat, origin_lon = denormalize_gps(idx[0]/grid_res, idx[1]/grid_res)

    # Format Google Maps Link
    maps_link = f"https://www.google.com/maps/search/?api=1&query={origin_lat},{origin_lon}"

    print(f"\n--- POLLUTION DETECTIVE REPORT ---")
    print(f"Peak Concentration: {np.max(field):.2f} µg/m³")
    print(f"Originator GPS: {origin_lat:.4f}, {origin_lon:.4f}")
    print(f"View on Map: {maps_link}")

    # Plotting
    plt.imshow(field.T, extent=(0,1,0,1), origin='lower', cmap='hot')
    plt.colorbar(label=r'PM2.5 ($\mu g/m^3$)')
    plt.scatter(sensor_coords[:,1], sensor_coords[:,0], c='white', label='Sensors')
    plt.title("Delhi-NCR Pollution Field Peak Detection")
    plt.show()
