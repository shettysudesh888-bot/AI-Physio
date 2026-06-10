# AI Physio: Comprehensive Technical Reference

This document serves as the complete technical blueprint for the **AI Physio** application. It details the entire architecture, technologies used, the machine learning pipeline, and the step-by-step data flow from the moment a user uploads an X-ray to the generation of a personalized recovery plan.

---

## 1. System Architecture & Tech Stack

The application is built on a modern, decoupled Client-Server architecture.

### **Frontend (Client-Side)**
- **HTML5 & CSS3:** Provides a responsive, accessible two-phase user interface. Uses CSS variables for theming and pure CSS animations (no external heavy frameworks like Bootstrap or Tailwind, ensuring maximum performance).
- **Vanilla JavaScript (`app.js`):** Handles all DOM manipulation, API routing, and state management without the overhead of React or Vue. Uses `sessionStorage` for temporary persistence.
- **FontAwesome:** Used for vector icons (via CDN).

### **Backend (Server-Side)**
- **Python 3.10+:** The core programming language.
- **FastAPI:** A high-performance web framework used to build the RESTful API endpoints. It handles concurrent requests asynchronously.
- **Pydantic:** Used heavily in `app.py` for strictly validating incoming data payloads (e.g., ensuring `pain_level` is an integer between 0-10).
- **SQLite3:** A lightweight, serverless relational database used to store User accounts, authentication Sessions, and historical Cases/Reports.
- **ReportLab:** A Python library used in `report.py` to dynamically generate downloadable PDF files for Recovery Plans and Nutrition Plans.

### **Machine Learning & AI**
- **PyTorch & Torchvision:** The deep learning framework powering the bone fracture classification model.
- **ResNet18:** The specific Convolutional Neural Network (CNN) architecture used. It was chosen for its excellent balance of high accuracy and fast inference speed.
- **Groq Cloud API (Llama 3):** Powers the conversational Medical Assistant, allowing lightning-fast, context-aware chatting.

---

## 2. The Machine Learning Pipeline

The machine learning core is split into two phases: Training (Notebooks) and Inference (Backend).

### **A. Training Pipeline (`notebooks/train_model.py`)**
1. **Architecture:** Uses a pre-trained ResNet18 model (transfer learning from ImageNet). The final fully connected layer is modified to output a single value for binary classification (Fracture vs. Normal).
2. **Data Augmentation:** To prevent overfitting, images are dynamically altered during training (Random Horizontal Flips, Random Rotations up to 15 degrees).
3. **Hyperparameters:** 
   - **Optimizer:** Adam (Adaptive Moment Estimation) with a learning rate of `1e-4`.
   - **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy), standard for binary classification.
   - **Scheduler:** `ReduceLROnPlateau` lowers the learning rate if the validation accuracy stops improving.
4. **Early Stopping:** The script monitors validation loss. If the loss fails to improve for 5 consecutive epochs, training halts early to prevent overfitting.
5. **Checkpointing:** The model strictly saves a `state_dict` (just the weights, which is safer and cleaner than saving the entire model object) to `backend/model/bone_fracture_resnet18_statedict.pth`.

### **B. Evaluation (`notebooks/evaluate_model.py`)**
This script tests the finalized model against unseen Test data, generating exact Accuracy, Precision, Recall, and F1-Score metrics. It outputs visual Confusion Matrices (`confusion_matrix_validation.png`, `confusion_matrix_test.png`).

### **C. Inference Pipeline (`backend/predict.py`)**
When the server boots, it loads the saved `state_dict` into an empty ResNet18 shell. 
When an image arrives:
1. It is converted to RGB.
2. Resized to `224x224` pixels.
3. Converted to a Tensor and normalized using standard ImageNet mean/std values.
4. Passed through the model. The raw "logit" output is passed through a Sigmoid function to get a probability between 0.0 and 1.0. A threshold of `0.5` determines if it is a fracture.

---

## 3. The Backend Engine & Logic Routing

The backend consists of several highly modularized files.

### **`app.py` (The Traffic Controller)**
Defines all API endpoints:
- `POST /upload`: Saves the raw X-ray image to a temporary local directory and returns a unique UUID.
- `POST /predict/{file_id}`: Runs the PyTorch inference pipeline and returns the AI's diagnosis (Fracture/Normal).
- `POST /recovery/{file_id}`: The core logic endpoint. It accepts the user's clinical intake form data, runs the diagnosis, and generates the final plan.
- `GET /cases` & `DELETE /cases/{case_id}`: Handles fetching and deleting historical reports from the SQLite database.

### **`recommendation.py` (The Exercise Dictionary)**
Contains a massive mapped dictionary (`EXERCISE_MAP`). It maps specific body parts (e.g., `shoulder`, `knee`) and specific recovery stages (e.g., `acute_phase`, `late_recovery`) to exact medical exercises. 

### **`recovery_guidance.py` (The Rule Engine)**
Takes the data from `recommendation.py` and the user's clinical context to build the final "Dos and Don'ts".
- **Dynamic Rules:** If `body_part` is "leg", it adds weight-bearing restrictions. If it's "wrist", it adds lifting restrictions.
- **Red Flags:** Monitors user inputs for dangerous combinations (e.g., high pain + severe swelling) and triggers medical alerts.
- **Exercise Eligibility:** A strict boolean gateway. If a user is in an acute fracture phase, `exercise_eligible` is set to `False`, and the exercise plan is safely blocked.

### **`llm_explanation.py` (The AI Chatbot)**
Hooks into the Groq API. It takes the *entire* patient context (Body Part, Stage, Model Prediction, Allowed Exercises) and injects it into a hidden System Prompt. This ensures the Llama 3 model acts strictly as a physiotherapist and knows the patient's exact medical state before answering any chat messages.

---

## 4. The Data Flow (Step-by-Step UX)

Here is exactly what happens when a user uses AI Physio:

1. **Upload:** User uploads an image. Frontend calls `/upload`. Backend saves it and returns `file_id`.
2. **Phase 1 (Detection):** Frontend calls `/predict`. Backend runs PyTorch ResNet18 and returns `fracture` (e.g., 98% confidence). UI updates.
3. **Clinical Intake:** User fills out the 8-field form (Body Part, Pain Level, Stage, etc.).
4. **Phase 2 (Recovery Generation):** User clicks "Generate". Frontend calls `/recovery`. 
   - Backend combines the PyTorch result with the 8 clinical fields.
   - `recovery_guidance.py` calculates eligibility.
   - `recommendation.py` fetches the correct exercises for that exact body part.
   - Backend saves the entire JSON object to the SQLite database.
5. **Rendering:** Frontend receives the JSON. It uses JavaScript DOM manipulation to build the Exercise Cards, Dos and Don'ts lists, and Nutrition tips dynamically on the screen.
6. **Chatting:** User clicks the Assistant button. Frontend calls `/assistant`. Backend passes the patient's context to Groq, Groq returns a personalized answer.
7. **Downloading:** User clicks "Download Report". Frontend calls `/report`. Backend uses `ReportLab` to draw a PDF file natively in Python and streams it back to the browser for download.
