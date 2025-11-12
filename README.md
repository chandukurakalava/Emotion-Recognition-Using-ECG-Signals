# Emotion Recognition in Electrocardiogram (ECG) Signals

This project presents a deep learning-based approach to detect and classify human emotions such as **joy, calmness, excitement, and tension** using ECG signals. The goal is to develop a real-time, non-intrusive emotion recognition system that can be integrated into healthcare, human-computer interaction, and wearable technologies.

---

## 📘 Abstract

Emotion-related changes in ECG signals are analyzed using advanced preprocessing, feature extraction, and classification methods. Discrete Cosine Transform (DCT) is used to extract meaningful features. Machine learning models like SVM, Random Forest, and KNN are evaluated. Particle Swarm Optimization (PSO) is applied to optimize the SVM parameters. Among the classifiers, SVM achieves the highest classification accuracy for detecting emotions.

**Keywords**: ECG, DCT, SVM, PSO, CNN, LSTM, Feature Extraction, Emotion Detection

---

## 🧠 Core Features

- ECG signal noise removal and preprocessing
- Feature extraction using DCT and Wavelet Transform
- Dual-mode classification using:
  - **CNN** for image-based ECG scalograms
  - **LSTM** for sequential ECG signal classification
- Parameter tuning using PSO for SVM
- Real-time and accurate detection of emotional states

---

## 🔧 Technologies Used

- **MATLAB 2013a**
- **Convolutional Neural Network (CNN)**
- **Long Short-Term Memory (LSTM)**
- **Support Vector Machine (SVM)**
- **Random Forest, K-Nearest Neighbor (KNN)**
- **Particle Swarm Optimization (PSO)**
- **Wavelet Transform & FIR Filters**
- **DREAMER Dataset**

---

## 🧪 Methodology

### 📊 Data Preprocessing
- Noise removal using Butterworth and FIR filters
- Wavelet decomposition for artifact removal
- Extraction of R-R intervals, BPM, QRS complex, etc.

### 🤖 Classification
- CNN: for scalogram/spectrogram image inputs
- LSTM: for sequential ECG signal data
- VGG-19 adapted for ECG image classification
- PSO for SVM hyperparameter tuning

### 📁 Dataset
- DREAMER Dataset (23 participants with emotion-labeled ECG)
- Augmented using time warping, noise injection, segmentation

---

## 📊 Simulation Results

- 82,000 samples
- 70% training, 20% validation, 10% testing split
- Accuracy evaluated for valence, arousal, dominance categories
- Observed training plateau after ~1000 epochs
- Models optimized using Adam optimizer

---

## ✅ Advantages

- High accuracy in stress and fear detection
- Works in real time
- Suitable for wearable device integration
- Uses non-intrusive ECG-based emotion sensing
- Extensible to multimodal input and animal emotion tracking

---

## 📍 Applications

- Mental Health Monitoring
- Personalized Healthcare & Therapy
- Human-Computer Interaction (HCI)
- Education & Adaptive Learning
- Workplace Well-being Systems
- Automotive Driver Monitoring
- Emotion-Aware Gaming & Media
- Consumer Behavior & Marketing
- Security and Surveillance
- Emotion-Enabled Social Robotics
- Sports & Athlete Stress Analysis

---

## 🔭 Future Scope

- Build a proprietary dataset using custom 3-lead ECG hardware
- Add fear classification in addition to stress
- Integrate additional sensors (EEG, GSR, respiration)
- Deploy edge computing models on wearables
- Add Explainable AI (XAI) interfaces
- Implement federated learning across multiple users

---

## 📂 Project Structure

```
AI-ECG-Emotion-Recognition/

📁 source code/
├── .git/                         # Git configuration folder
├── F001                          # Text Document
├── F002                          # Text Document
├── F003                          # Text Document
├── F010                          # Text Document
├── ✅ filename.mat               # MATLAB data file (used for model/data storage)
├── Final report.pdf              # Project report (PDF)
├── mainfornormal.m               # Main MATLAB script (entry point for execution)
├── Person_01                     # Likely raw data or configuration file (Text)
├── ✅ s1.mat                     # MATLAB data file (processed ECG or features)

```

---

## 📚 References

- DREAMER Dataset for emotion-labeled ECG
- “Discrete Cosine Transform” – N. Ahmed et al.
- “Support Vector Machines” – Cortes & Vapnik
- “Wavelet ECG Feature Detection” – Li et al.
- “Particle Swarm Optimization” – Kennedy (2011)
- More listed in `Final_Report.pdf` under References

---

## 👨‍🎓 Project Info

> Developed by:  
> - **Kummara Balamanikanta** – 22F65A0402  
> - **S. Dilip Chowdary** – 21F61A0438  
> - **Kurakalava Chandu** – 21F61A0429  
> - **Vootla Eshwar Sai** – 21F61A0444  
> - **Tirunam Guravaiah** – 21F61A0451  
>
> Final Year B.Tech ECE Students  
> Department of Electronics and Communications Engineering  
> Siddharth Institute of Engineering & Technology (SIETK), Puttur  
> Guided by: **Mr. C. Vijaya Bhaskar**, Associate Professor  
> Academic Year: **2024–2025**

