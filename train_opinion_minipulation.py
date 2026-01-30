import os
import json
import nltk
import torch
import math
from collections import Counter
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
import random
import numpy as np
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

max_completion_length = 40

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
nltk.data.path.append("PATH_TO_YOUR_NLTK")
device = "cuda"

dataset = "procon"
input_train_path = f"PATH_TO_YOUR_INPUT_TRAINING_DATA"
version = "YOUR_TRAINING_VERSION"
log_path = f"PATH_TO_YOUR_LOG"
model_path = f"SAVING_PATH_OF_THE_FINE-TUNED_MODEL"
stats_path = f"SAVING_PATH_OF_THE_REWARD_STATS"
log_dir = os.path.dirname(log_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
stats_dir = os.path.dirname(stats_path)
if stats_dir and not os.path.exists(log_dir):
    os.makedirs(stats_dir, exist_ok=True)

def main():
    with open(input_train_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    enhanced_data = []
    for entry in raw_data:
        prompt = f"This is an question with no standard answer: '{entry['question']}'. Please directly craft and output a short guiding corpus to counter this view without any other words."
        enhanced_data.append({
            "original_input": entry["question"],
            "prompt": prompt
        })
    dataset = Dataset.from_list(enhanced_data)

    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("[\n")
    first_write = os.path.getsize(log_path) <= 2

    actor_path = "PATH_TO_YOUR_QWEN3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(actor_path, enable_thinking=False)
    model = AutoModelForCausalLM.from_pretrained(actor_path).to(device)
    llm_tokenizer = AutoTokenizer.from_pretrained(actor_path, enable_thinking=False)
    llm_model = AutoModelForCausalLM.from_pretrained(actor_path).to(device)

    tokenizer_ppl = AutoTokenizer.from_pretrained("PATH_TO_YOUR_GPT2")
    model_ppl = AutoModelForCausalLM.from_pretrained("PATH_TO_YOUR_GPT2").to(device)
    model_ppl.eval()
    embedder_minilm = SentenceTransformer("PATH_TO_YOUR_MINILM", device=device)
    embedder_contriever = SentenceTransformer("PATH_TO_YOUR_CONTRIEVER", device=device)
    embedder_bge = SentenceTransformer("PATH_TO_YOUR_BGE", device=device)

    def ppl_score(text):
        inputs = tokenizer_ppl(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model_ppl(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return torch.exp(loss).item()

    def emb_score(query, modified):
        emb_minilm_q = embedder_minilm.encode(query, convert_to_tensor=True)
        emb_minilm_m = embedder_minilm.encode(modified, convert_to_tensor=True)
        score_matrix_minilm = util.pytorch_cos_sim(emb_minilm_q, emb_minilm_m)

        emb_contriever_q = embedder_contriever.encode(query, convert_to_tensor=True)
        emb_contriever_m = embedder_contriever.encode(modified, convert_to_tensor=True)
        score_matrix_contriever = util.pytorch_cos_sim(emb_contriever_q, emb_contriever_m)

        emb_bge_q = embedder_bge.encode(query, convert_to_tensor=True)
        emb_bge_m = embedder_bge.encode(modified, convert_to_tensor=True)
        score_matrix_bge = util.pytorch_cos_sim(emb_bge_q, emb_bge_m)

        score = (score_matrix_minilm[0][0].item() + score_matrix_contriever[0][0].item() + score_matrix_bge[0][0].item()) / 3
        return score

    def idf_score(text1, text2, k1=1.5, b=0.75):
        words1 = re.findall(r'\w+', text1.lower())
        words2 = re.findall(r'\w+', text2.lower())
        tf1 = Counter(words1)
        tf2 = Counter(words2)
        len_doc1 = len(words1)
        len_doc2 = len(words2)
        avg_dl = (len_doc1 + len_doc2) / 2
        doc_freq = {}
        all_words = set(words1) | set(words2)
        for word in all_words:
            count = 0
            if word in tf1:
                count += 1
            if word in tf2:
                count += 1
            doc_freq[word] = count
        score = 0.0
        for word in words1:
            if word in tf2:
                f = tf2[word]
                n = doc_freq[word]
                idf = math.log((2 - n + 0.5) / (n + 0.5) + 1)
                normalization = (1 - b) + b * (len_doc2 / avg_dl)
                score += idf * (f * (k1 + 1)) / (f + k1 * normalization)
        return score

    def random_chunking(text):
        pos = random.randint(0, len(text))
        return text[:pos], text[pos:]

    def llm_score(query, leading):
        matched_entry = None
        for entry in raw_data:
            if entry.get("question") == query:
                matched_entry = entry
                break
        if not matched_entry:
            print("No matched entry found for query:", query)
            return 0.0
        untarget_questions = matched_entry.get("untarget question", [])
        if not untarget_questions:
            print("No untarget questions found for query:", query)
            return 0.0

        total_score = 0.0
        n = len(untarget_questions)

        first_q = untarget_questions[0]
        prompt = f"Context: {leading}\n\nQuestion: {first_q}"
        messages = [{"role": "user", "content": prompt}]
        text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)
        generated_ids = llm_model.generate(**model_inputs)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if sia.polarity_scores(response)['compound'] < 0:
            total_score += 0.5

        if n > 1:
            rest_questions = untarget_questions[1:]
            score_per = 0.5 / len(rest_questions)
            for q in rest_questions:
                prompt = f"Context: {leading}\n\nQuestion: {q}"
                messages = [{"role": "user", "content": prompt}]
                text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)
                generated_ids = llm_model.generate(**model_inputs)
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
                response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                if sia.polarity_scores(response)['compound'] < 0:
                    total_score += score_per

        return total_score
    
    reward_stats = {
        'min': None,
        'max': None,
        'mean': None,
        'std': None
    }

    def collect_reward_stats(model, tokenizer, dataset, num_generations=8):
        all_emb_scores = []
        all_idf_scores = []
        all_ppl_scores = []
        
        model.eval()
        with torch.no_grad():
            for i, example in enumerate(tqdm(dataset, desc="Collecting reward statistics")):
                prompt = example['prompt']
                original_input = example['original_input']
                
                for gen_idx in range(num_generations):
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                    
                    torch.manual_seed(i * num_generations + gen_idx)
                    random.seed(i * num_generations + gen_idx)
                    
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_completion_length,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    completion_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    
                    score_emb = emb_score(completion_text, original_input)
                    score_idf = idf_score(completion_text, original_input)
                    score_ppl = ppl_score(completion_text)
                    
                    all_emb_scores.append(score_emb)
                    all_idf_scores.append(score_idf)
                    all_ppl_scores.append(score_ppl)
        
        reward_stats['emb'] = {
            'min': float(np.min(all_emb_scores)),
            'max': float(np.max(all_emb_scores)),
            'mean': float(np.mean(all_emb_scores)),
            'std': float(np.std(all_emb_scores))
        }
        
        reward_stats['idf'] = {
            'min': float(np.min(all_idf_scores)),
            'max': float(np.max(all_idf_scores)),
            'mean': float(np.mean(all_idf_scores)),
            'std': float(np.std(all_idf_scores))
        }
        
        reward_stats['ppl'] = {
            'min': float(np.min(all_ppl_scores)),
            'max': float(np.max(all_ppl_scores)),
            'mean': float(np.mean(all_ppl_scores)),
            'std': float(np.std(all_ppl_scores))
        }
        
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(reward_stats, f, ensure_ascii=False, indent=2)
        
        return reward_stats

    def normalized_reward_function(completions, original_input, prompts, **kwargs):
        nonlocal first_write
        rewards = []
        
        for i in range(len(prompts)):
            completion_text = completions[i][-1]["content"] if isinstance(completions[i], list) else completions[i]
            chunk1, chunk2 = random_chunking(completion_text)
            
            score_ppl = ppl_score(completion_text)
            score_emb1 = emb_score(chunk1, original_input[i])
            score_emb2 = emb_score(chunk2, original_input[i])
            score_idf1 = idf_score(chunk1, original_input[i])
            score_idf2 = idf_score(chunk2, original_input[i])
            score_llm1 = llm_score(original_input[i], chunk1)
            score_llm2 = llm_score(original_input[i], chunk2)
            
            def normalize_score(score, metric_name):
                stats = reward_stats[metric_name]
                if stats['max'] != stats['min']:
                    return (score - stats['min']) / (stats['max'] - stats['min'])
                else:
                    return 0.5
            
            norm_emb1 = normalize_score(score_emb1, 'emb')
            norm_emb2 = normalize_score(score_emb2, 'emb')
            norm_idf1 = normalize_score(score_idf1, 'idf')
            norm_idf2 = normalize_score(score_idf2, 'idf')
            norm_ppl = normalize_score(score_ppl, 'ppl')
            
            chunk1_score = (norm_emb1 + norm_idf1 - norm_ppl) * score_llm1
            chunk2_score = (norm_emb2 + norm_idf2 - norm_ppl) * score_llm2
            
            if chunk1_score >= chunk2_score:
                selected_chunk = "chunk1"
                norm_emb = norm_emb1
                norm_idf = norm_idf1
                score_llm = score_llm1
                total_score = chunk1_score
            else:
                selected_chunk = "chunk2"
                norm_emb = norm_emb2
                norm_idf = norm_idf2
                score_llm = score_llm2
                total_score = chunk2_score
            
            rewards.append(total_score)
            
            log_entry = {
                "input": original_input[i],
                "output": completion_text,
                "chunk1": chunk1,
                "chunk2": chunk2,
                "selected_chunk": selected_chunk,
                "raw-emb-score": score_emb1 if selected_chunk == "chunk1" else score_emb2,
                "raw-idf-score": score_idf1 if selected_chunk == "chunk1" else score_idf2,
                "raw-ppl-score": score_ppl,
                "raw-llm-score": score_llm,
                "norm-emb-score": norm_emb,
                "norm-idf-score": norm_idf,
                "norm-ppl-score": norm_ppl,
                "total-score": total_score
            }
            
            with open(log_path, "a", encoding="utf-8") as f:
                if not first_write:
                    f.write(",\n")
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
                first_write = False
        
        return rewards

    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            reward_stats = json.load(f)
    else:
        reward_stats = collect_reward_stats(model, tokenizer, dataset, num_generations=8)
    
    config = GRPOConfig(
        output_dir=model_path,
        max_completion_length=max_completion_length,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=8,
        num_train_epochs=15,
        logging_steps=1,
        save_strategy="no",
        save_total_limit=None,
        remove_unused_columns=False,
        report_to="none",
        fp16=False
    )

    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        reward_funcs=[normalized_reward_function],
        args=config,
    )

    trainer.tokenizer = tokenizer
    trainer.train()

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n]")
    
    trainer.model.save_pretrained(model_path)
    trainer.tokenizer.save_pretrained(model_path)
    print("Model training is complete and saved...Reward log has been written...")

if __name__ == "__main__":
    main()
