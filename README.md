

# 🌫️ Delhi-NCR Pollution Field Reconstruction & Source Detection

A **physics-informed neural field model** that reconstructs the **PM2.5 pollution distribution across Delhi-NCR** using sparse sensor readings and estimates the **probable origin of peak pollution concentration**.

The model combines **neural networks + physical diffusion constraints** to infer pollution levels in areas where sensors are not present.

---

# 📊 Project Result

![Pollution Field](result.png)

### Pollution Detective Report

| Metric | Value |
|------|------|
Peak Concentration | **338.98 µg/m³**
Estimated Origin | **28.7200, 77.3920**
Map Link | https://www.google.com/maps/search/?api=1&query=28.72,77.392

---

# 🚀 Motivation

Air pollution sensors are **sparse and expensive**, meaning cities only measure pollution at a few locations.

This creates a problem:

• Large areas of the city have **no direct measurements**  
• Pollution sources are **difficult to locate**  
• Environmental monitoring becomes **incomplete**

This project explores:

**Can we reconstruct the full pollution field of a city using only a few sensors?**

And further:

**Can we estimate where the pollution likely originated?**

---

# 🧠 Core Idea

We model pollution as a **continuous spatial field**

PM(x, y)

A neural network learns this function so it can predict pollution levels at **any coordinate in the city**.

However, instead of learning purely from data, we enforce a **physical constraint**:

Pollution behaves like **diffusion**, meaning the field should remain **smooth and continuous**.

Loss function used:

Total Loss = Data Loss + λ × Physics Loss

### Data Loss

Matches predicted pollution values to real sensor readings.

MSE(predicted_sensor, actual_sensor)

### Physics Loss

Encourages the field to behave like diffusion by minimizing the gradient magnitude.

‖∇PM(x,y)‖²

This makes the pollution field **physically realistic instead of noisy interpolation**.

---

# 🏗 Model Architecture

Input: latitude and longitude coordinates

Neural Field Network

2 → 128 → 128 → 1

Activation functions:

Tanh

The network outputs:

PM2.5 concentration for any spatial coordinate.

---

# 📍 Sensor Data Used

Example sensor readings used for reconstruction.

| Location | PM2.5 |
|--------|------|
Anand Vihar | 460 |
Loni (Hotspot) | 510 |
ITO | 195 |
RK Puram | 169 |
Sonia Vihar | 169 |

Coordinates are normalized before training.

---

# 🔬 Training Process

1. Normalize GPS coordinates of sensors  
2. Create a **city grid (50×50)**  
3. Train neural field to match sensor readings  
4. Apply physics-based smoothness constraint  
5. Reconstruct full pollution field  
6. Detect highest pollution location  

Training parameters:

Epochs: 1000  
Optimizer: Adam  
Learning Rate: 0.005  
Grid Resolution: 50×50

---

# 📈 Output

The system produces:

• A **PM2.5 heatmap across Delhi-NCR**  
• Estimated **pollution source location**  
• A **Google Maps link** for the detected origin  

Example output:

--- POLLUTION DETECTIVE REPORT ---

Peak Concentration: 338.98 µg/m³  
Originator GPS: 28.7200, 77.3920  

View on Map  
https://www.google.com/maps/search/?api=1&query=28.72,77.392

---

# 🗺 Visualization

White dots represent **sensor locations**.

The heatmap represents the **reconstructed pollution field**.

Brighter areas correspond to **higher PM2.5 concentration**.





# 💡 Potential Applications

Environmental monitoring  
Smart city infrastructure  
Pollution source detection  
Urban planning  
Air quality prediction  

