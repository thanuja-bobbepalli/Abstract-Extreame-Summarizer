# Scientific Abstract Extreme Summarization using a GPT-2 Transformer 

A Transformer-based **research paper abstract summarization system** built using a GPT-style architecture implemented from scratch in **PyTorch**.  
The model generates concise summaries from long scientific abstracts and is deployed using **Hugging Face Spaces** for interactive inference.

---

## 🚀 Live Demo

You can try the model directly using the Hugging Face interface.

### Steps to run the demo

1. Open the Hugging Face Space link below  
2. Paste a research abstract in the input box  
3. Click **Generate Summary**  
4. The model will generate a concise summary  

👉 **Live Demo:**  
https://huggingface.co/spaces/Thanuja-Bobbepalli/summarizer
demo : <img width="1902" height="827" alt="image" src="https://github.com/user-attachments/assets/157bee54-f936-4a91-965f-26d992d09f82" />

---

## 🧠 Model Overview

This project implements a **GPT-style Transformer model from scratch** and fine-tunes it for research abstract summarization.

Key components:

- Token Embedding Layer
- Positional Embedding Layer
- Multi-Head Self Attention
- Feed Forward Network
- Layer Normalization
- Transformer Blocks
- Language Modeling Head

---

## 📊 Example Output

### Input Abstract

```
In this paper, we consider Nesterov's Accelerated Gradient method for solving nonlinear inverse and ill-posed problems. Known to be a fast gradient-based iterative method for solving well-posed convex optimization problems, this method also leads to promising results for ill-posed problems.
```

### Ground Truth Summary

```
Nesterov's Accelerated Gradient Method for Nonlinear Ill-Posed Problems
```

### Generated Summary

```
A method for accelerated gradient optimization applied to nonlinear inverse problems.
```

---

## 📈 Evaluation Metrics

Model performance is evaluated using **ROUGE metrics**.

Example results:

```
ROUGE-1 : 0.185
ROUGE-2 : 0.071
ROUGE-L : 0.168
```

---

## 📂 Project Structure

```
Abstract-Extreme-Summarizer
│
├── model.py
├── generate.py
├── tokenizer_utils.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 🛠 Technologies Used

- Python
- PyTorch
- Transformer Architecture
- GPT-style Language Modeling
- Hugging Face Spaces
- Natural Language Processing

---

## 👤 Author

**Thanuja Bobbepalli**

Machine Learning enthusiast focused on:

- Natural Language Processing
- Transformer Architectures
- Large Language Models
