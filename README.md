# Confundo

> **Confundo: Learning to Generate Robust Poison for Practical RAG**

---

## 📖 Abstract

Retrieval-augmented generation (RAG) systems are increasingly deployed to enhance the capability of large language models with external knowledge, yet their reliance on untrusted data sources renders them vulnerable to poisoning attacks. Existing RAG poisoning methods remain limited in practice: they are designed for a single attack objective, rely on unrealistic assumptions, or fail to withstand essential preprocessing steps such as tokenization and chunking. In this paper, we present Confundo, a unified learning-based framework that generates robust poison texts to effectively compromise practical RAG systems. Confundo is explicitly optimized to be an end-to-end, pipeline–aware framework, ensuring that the generated poison texts remain effective across diverse and unknown ingestion, retrieval, and generation configurations. Within a single framework, Confundo supports multiple attack objectives, including targeted factual correctness manipulation, opinion manipulation, and hallucination amplification. Extensive experiments across multiple datasets, RAG configurations, and defenses show that Confundo consistently outperforms state-of-the-art attacks by large margins. We further present a case study on protecting web content from unauthorized use by RAG systems, demonstrating how Confundo can be repurposed as a defensive mechanism against large-scale web scraping. Our findings reveal a critical gap between existing defenses and the growing practicality of robust RAG poisoning, underscoring the urgent need for principled, pipeline-aware security mechanisms for real-world RAG deployments.

---

## 🏗️ Project Structure

```
Confundo-main/
├── README.md                                      # Project documentation
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

