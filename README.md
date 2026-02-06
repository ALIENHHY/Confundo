# Confundo

> **Confundo: Learning to Generate Robust Poison for Practical RAG**

---

## 🏗️ Project Structure

```
Confundo-main/
├── README.md                                      # Project documentation
├── Examples/                                      # Prompt optimization (conjunction selection + GRPO training)
│   ├── factual_correctness_manipulation.json      # Examples of factual correctness manipulation attack
│   ├── opinion_manipulation.json                  # Examples of opinion manipulation attack
│   ├── hallucination_amplification.json           # Examples of hallucination amplification attack
├── train_factual_correctness_manipulation.py      # RL fine-tuning for factual correctness manipulation
├── train_opinion_minipulation.py                  # RL fine-tuning for opinion minipulation
├── train_hallucination_amplification.py           # RL fine-tuning for hallucination amplification
├── rag_pipeline.py                                # Evaluation on RAG pipeline
```

---

## 🚀 Usage

### 1. Installation
```bash
git clone https://github.com/ALIENHHY/Confundo.git
cd Confundo-main
pip install -r requirements.txt
```

### 2. GRPO Training

```bash
python train_factual_correctness_manipulation.py
```

```bash
python train_opinion_minipulation.py
```

```bash
python train_hallucination_amplification.py
```

### 3. Evaluate attacks
```bash
python rag_pipeline.py
```

---

## 📊 Datasets

We use five publicly available datasets to evaluate GhostRAG:

### 1. Factual Correctness Manipulation
- **Harry Potter:** [[link]](https://huggingface.co/datasets/vapit/HarryPotterQA)
- **NewsQA:** [[link]](https://huggingface.co/datasets/glnmario/news-qa-summarization)
- **OCRBench:** [[link]](https://huggingface.co/datasets/opendatalab/OHR-Bench)

### 2. Opinion Manipulation
- **PROCON:** Collected by [Topic-FlipRAG](https://www.usenix.org/system/files/usenixsecurity25-gong-yuyang.pdf) and [FlippedRAG](https://dl.acm.org/doi/10.1145/3719027.3765023) from [PROCON.ORG](https://testing-www.procon.org/)

### 3. Hallucination Amplification
- **RAGTruth:** [[link]](https://github.com/ParticleMedia/RAGTruth)

