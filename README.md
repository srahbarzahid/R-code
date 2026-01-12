

# 🧠 Expert System – A No-Code UI for Machine Learning

## 📌 Overview

**Expert System** is a web-based, no-code machine learning platform designed to simplify the entire machine learning workflow. It enables users to build, train, visualize, evaluate, and deploy machine learning models **without writing any code**.

The system focuses on improving the **workflow of model building** by offering an intuitive graphical interface that removes the complexity of traditional ML frameworks. It is especially useful for beginners and non-technical users who want hands-on exposure to machine learning concepts while clearly understanding each step of the process.

---

## 🚀 Problem Statement

Most existing machine learning tools and frameworks require strong programming knowledge and familiarity with complex libraries. This creates a barrier for many users who want to apply machine learning but lack coding expertise.

There is a need for a **no-code machine learning system** that:

* Simplifies the ML workflow
* Eliminates dependency on programming
* Enables experimentation with multiple algorithms
* Provides learning support alongside implementation

---

## 🌍 Why This Idea Matters

Machine learning adoption is growing rapidly, but **model building remains inaccessible** to a large section of users due to steep technical requirements. Many platforms either focus on **automation without understanding** or remain **code-heavy and complex**.

**Expert System prioritizes workflow clarity over automation**, ensuring users understand *what happens at each stage* of the ML pipeline while still remaining completely no-code. This makes the platform suitable not only for building models, but also for **learning how machine learning works in practice**.

---

## 🎯 Objectives

* Allow users to upload datasets and preprocess data easily
* Provide access to multiple machine learning algorithms through a UI
* Enable model training, evaluation, and prediction without coding
* Provide interactive visualizations for better understanding of results
* Help users understand ML concepts through guided walkthroughs
* Allow trained models to be exported for real-world use

---

## ✨ What Makes Expert System Different

Unlike traditional AutoML tools or notebook-based environments, **Expert System is workflow-oriented**.

It:

* Focuses on **step-by-step ML workflow transparency**
* Avoids black-box automation
* Encourages **guided decision-making** instead of one-click results
* Combines **education and experimentation** in a single interface
* Allows users to visually follow the **data → model → result** flow

The goal is not just faster models, but **clearer understanding**.

---

## 🧪 Real-World Use Case Example

**Scenario:**
A college student has a CSV dataset and wants to predict student performance but has no coding experience.

**Using Expert System:**

1. Uploads the dataset
2. Cleans and preprocesses data using UI controls
3. Selects a classification algorithm
4. Visualizes correlations and clusters
5. Trains the model and evaluates accuracy
6. Exports the trained model for future predictions

All steps are completed **without writing code**, while still understanding *why* each step exists.

---

## 🤖 Role of AI in the System

AI is used as an **enabler**, not a replacement for learning.

* Assists in algorithm selection guidance
* Helps interpret visualizations and model behavior
* Provides contextual hints and explanations
* Accelerates experimentation without hiding logic

The focus is to **speed up understanding**, not just outcomes.

---

## 🧩 Features

### 🔹 User Module

* User authentication (Signup & Login)
* Dataset upload and management
* Data preprocessing:

  * Handling missing values
  * Encoding categorical data
  * Feature scaling
* Model training:

  * Classification
  * Regression
  * Clustering
* Interactive visualizations:

  * Heatmaps
  * Scatter plots
  * Cluster visualizations
* Prediction using trained models
* Export and download trained models

---

### 🔹 Admin Module

* Secure authentication using JWT
* Manage user credentials
* Manage stored and exported models

---

### 🔹 Learning Module

* Learning pages for beginners
* Quick reference notes
* Guided tours for understanding ML workflows

---

## 📊 Machine Learning Capabilities

* Supports **7 machine learning algorithms**
* Covers:

  * Classification
  * Regression
  * Clustering
* Customizable train–test split
* Preloaded datasets for experimentation
* Model evaluation and result visualization

---

## 🛠️ Technologies Used

### Frontend

* HTML5
* CSS3
* Bootstrap 5
* JavaScript

### Backend

* Python
* Django Framework
* Scikit-learn
* Pandas & NumPy

### Visualization & UI Libraries

* D3.js
* Plotly.js
* Prism.js
* Intro.js

---

## 🏗️ Workflow Overview

1. Upload dataset
2. Preprocess data using UI
3. Select machine learning algorithm
4. Train and evaluate model
5. Visualize results
6. Make predictions
7. Export trained model

---

## 🎯 Target Users

* Beginners exploring machine learning
* Students learning ML concepts
* Non-technical users
* Anyone looking to build ML models without coding

---

## 🔮 Future Vision

Expert System can evolve into a complete **ML workflow assistant** by adding:

* Smart preprocessing recommendations
* Model comparison dashboards
* Dataset quality analysis
* Collaborative model building
* Cloud-based deployment support

The long-term vision is to make **machine learning approachable, explainable, and workflow-driven**.



## ⚙️ Installation & Setup

Clone the project:

```
git clone "repository link"

```

Navigate to the project directory:

```bash
cd expertsystem-main
```

Create a virtual environment:

```bash
python -m venv myenv
```

Activate the virtual environment:

**Linux / macOS**

```bash
source myenv/bin/activate
```

**Windows**

```bash
.\myenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run database migrations:

```bash
python manage.py migrate
```

Start the Django development server:

```bash
python manage.py runserver
```

