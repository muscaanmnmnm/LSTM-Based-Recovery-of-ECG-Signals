# LSTM-Based Recovery of ECG Signals

This project implements a Deep Learning model using **Long Short-Term Memory (LSTM) Autoencoders** to reconstruct and recover Electrocardiogram (ECG) signals. It is designed to handle "broken" or masked signals, effectively restoring missing data segments to aid in accurate medical diagnosis.

## 📌 Project Overview
In medical telemetry, ECG signals can often be corrupted or lost due to transmission errors or sensor malfunctions. This project builds a sequence-to-sequence autoencoder that learns the temporal structure of healthy heartbeats. It can take a signal with missing segments (simulated data loss) and reconstruct the original waveform.

**Key Features:**
* **Data Preprocessing:** specific reshaping of 1D signals for LSTM ingestion.
* **Architecture:** A compressed representation (Encoder-Decoder) approach.
* **Simulation:** Artificial masking of signal regions to test recovery capabilities.
* **Evaluation:** Visual comparison and Root Mean Square (RMS) error calculation.

## 📂 Dataset
The project uses the **MIT-BIH Arrhythmia Dataset**, sourced via the [Kaggle Heartbeat Dataset](https://www.kaggle.com/shayanfazeli/heartbeat).

* **Train Data:** `mitbih_train.csv`
* **Test Data:** `mitbih_test.csv`
* **Input Shape:** Each sample consists of 187 time steps representing a heartbeat.

## 🛠️ Methodology

### 1. Data Preparation
The raw CSV data is loaded and split into signals (features) and labels. [cite_start]The signals are reshaped from 2D arrays `(samples, 187)` to 3D arrays `(samples, 187, 1)` to satisfy LSTM input requirements[cite: 54].

### 2. Model Architecture
[cite_start]The model is built using **TensorFlow/Keras** with a Sequential API[cite: 65]. It follows an Autoencoder structure:

* **Input Layer:** `(187, 1)`
* **Encoder:**
    * `LSTM (64 units)`: Captures high-level temporal features.
    * [cite_start]`LSTM (32 units)`: Compresses the data into a lower-dimensional latent space[cite: 69].
* **Decoder:**
    * `LSTM (64 units)`: Expands the latent features back to the original sequence length.
* **Output:**
    * [cite_start]`TimeDistributed(Dense(1))`: Projects the output back to a single amplitude value per time step[cite: 71].

### 3. Training
[cite_start]The model is trained to minimize **Mean Squared Error (MSE)**, comparing the output directly to the input (reconstruction task)[cite: 73].
* **Epochs:** 20
* **Batch Size:** 32
* **Optimizer:** Adam

### 4. Testing & Recovery Simulation
To evaluate the model, we simulate data loss on the test set:
1.  [cite_start]**Masking:** A contiguous block of the signal (indices 100 to 137, approx. 20%) is forced to zero[cite: 111].
2.  **Prediction:** The masked signal is fed into the trained autoencoder.
3.  **Comparison:** The model's "Recovered" output is compared against the "Original" clean signal.

## 💻 Tech Stack
* **Python 3**
* **TensorFlow / Keras** (Deep Learning)
* **Pandas & NumPy** (Data Manipulation)
* **Matplotlib** (Visualization)
* **Scikit-Learn** (Metrics)

## 📊 Results
The notebook visualizes the performance by plotting three waveforms on the same graph:
1.  🟢 **Original Signal:** The ground truth.
2.  🔴 **Masked Input:** The broken signal fed to the model (dashed line).
3.  🔵 **Recovered Signal:** The model's reconstruction attempt.

[cite_start]Performance is quantified using **Root Mean Square Error (RMSE)** for each sample[cite: 125].

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pandas tensorflow matplotlib scikit-learn
    ```
3.  **Run the notebook:**
    Open the `.ipynb` file in Jupyter Notebook, Google Colab, or Kaggle and execute the cells sequentially.

---
*This project was developed as part of an investigation into signal processing and deep learning applications in healthcare.*
