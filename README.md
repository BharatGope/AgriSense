<h1 align="center">🌾 AgriSense – AI Crop Health Detector (Concept Prototype)</h1>

<p align="center">
  <b>AI-powered crop disease detection system built using PyTorch and Streamlit.</b><br>
</p>

---

## 🧠 Overview

**AgriSense** is a concept-stage AI prototype designed to assist farmers in identifying crop diseases early and receiving actionable recommendations.  
It uses **computer vision** and **deep learning** to analyze crop leaf images and classify potential diseases.

> ⚠️ *Note:* This repository currently represents a **concept demo**.  
> The model is not yet trained on agricultural datasets — it uses simulation logic for realistic behavior.  
> A fully trained model on the **PlantVillage dataset** is required for production use.

---

## 🚀 Features

- 🖼️ Upload a crop leaf image for instant analysis  
- 🔍 Simulated AI predictions with confidence scores  
- 💡 Smart treatment recommendations for each detected disease  
- 🌐 Runs fully offline using **Streamlit**  
- 🧩 Modular and ready for model integration  

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Backend | PyTorch, Torchvision |
| Language | Python 3.13+ |
| Data Source | PlantVillage (for model training) |
| Framework | ResNet18 (ImageNet weights for demo) |

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/BharatGope/AgriSense.git
   cd AgriSense

2. **Install Dependencies**

   ```bash
   pip install torch torchvision streamlit pillow
   ```

3. **Run the Application**

   ```bash
   python -m streamlit run demo.py
   ```

4. **Access the App**

   ```
   http://localhost:8501
   ```

---

## 🧠 Model Training (Required for Full Prototype)

The current prototype uses a **placeholder ResNet18** model trained on ImageNet for demonstration.
To make it production-ready:

1. Download the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).
2. Fine-tune a CNN model (e.g., ResNet18 or EfficientNet).
3. Save your trained weights as `plantvillage_model.pth`.
4. Update the app:

   ```python
   model.load_state_dict(torch.load("plantvillage_model.pth", map_location="cpu"))
   model.eval()
   ```

Once trained, the system can provide accurate disease classification across 38+ crop types.

---

## 🌱 Example Output (Concept Simulation)

| Input Image                    | Predicted Output        | Recommendation                                   |
| ------------------------------ | ----------------------- | ------------------------------------------------ |
| Tomato leaf with blight spots  | `Tomato___Late_blight`  | Apply copper-based fungicide and ensure airflow. |
| Potato leaf with brown lesions | `Potato___Early_blight` | Avoid overhead irrigation and rotate crops.      |
| Corn leaf with rust patches    | `Corn___Common_rust`    | Use rust-resistant corn varieties.               |
| Apple leaf with black spots    | `Apple___Black_rot`     | Remove infected leaves and spray fungicide.      |

---

## 🔮 Future Scope

* ☁️ Integrate with **AWS SageMaker** for scalable model training and deployment
* 🛰️ Add **IoT-based weather and soil sensors** for contextual insights
* 🌐 Build a multilingual mobile interface for farmers
* 🤖 Use **AWS Bedrock** for Generative AI explanations and **Guardrails** for responsible AI

---

## 📘 License

Licensed under the **Apache License 2.0**.
You may use, modify, and distribute this project in compliance with the license terms.

---

<p align="center">
  Developed with ❤️ by <b>Bharat Kumar Gope</b><br>
  <a href="https://github.com/BharatGope">GitHub Profile</a> • <i>Ramgarh Engineering College, CSE</i>
</p>

---
