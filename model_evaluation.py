from matplotlib import cm
import torch
import numpy as np
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Download required NLTK resources
try:
    nltk.download('wordnet')
    nltk.download('punkt')
except:
    print("NLTK download failed. If meteor score fails, install these manually.")

def calculate_metrics(reference, candidate):
    """Calculate BLEU, METEOR, and ROUGE scores"""
    # BLEU score calculation with smoothing
    ref_tokens = [reference.split()]
    cand_tokens = candidate.split()
    smooth = SmoothingFunction().method1
    bleu1 = sentence_bleu(ref_tokens, cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    # METEOR score
    try:
        meteor = meteor_score([reference.split()], candidate.split())
    except:
        meteor = 0
    
    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(reference, candidate)
    
    return {
        'bleu-1': bleu1,
        'bleu-4': bleu4,
        'meteor': meteor,
        'rouge-1': rouge['rouge1'].fmeasure,
        'rouge-2': rouge['rouge2'].fmeasure,
        'rouge-l': rouge['rougeL'].fmeasure
    }

def evaluate_model(model, tokenizer, test_df, device, num_examples=5, max_length=128):
    """Evaluate model and return metrics"""
    model.eval()
    all_metrics = []
    generated_texts = []
    
    # Process each example
    for idx in tqdm(range(len(test_df)), desc="Evaluating"):
        formal_text = test_df['formal'].iloc[idx]
        reference_informal = test_df['informal'].iloc[idx]
        
        # Prepare input
        input_text = f"convert to informal: {formal_text}"
        input_encoding = tokenizer(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate metrics
        metrics = calculate_metrics(reference_informal, generated_text)
        all_metrics.append(metrics)
        
        # Store generated text for display
        generated_texts.append({
            'formal': formal_text,
            'reference_informal': reference_informal,
            'generated_informal': generated_text
        })
    
    # Calculate average metrics
    avg_metrics = {metric: np.mean([item[metric] for item in all_metrics]) for metric in all_metrics[0].keys()}
    
    # Display random examples
    print("\n--- Sample Results ---")
    for i in np.random.choice(len(generated_texts), min(num_examples, len(generated_texts)), replace=False):
        print(f"\nExample {i+1}:")
        print(f"Formal: {generated_texts[i]['formal']}")
        print(f"Reference Informal: {generated_texts[i]['reference_informal']}")
        print(f"Generated Informal: {generated_texts[i]['generated_informal']}")
    
    # Display metrics
    print("\n--- Evaluation Metrics ---")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(generated_texts)
    results_df.to_csv("generation_results.csv", index=False)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([avg_metrics])
    metrics_df.to_csv("evaluation_metrics.csv", index=False)
    
    values = list(avg_metrics.values())
    # Normalize the number of bars to 0–1 range for the colormap
    norm = plt.Normalize(min(values), max(values))

    # Choose a colormap
    cmap = cm.viridis  # You can use 'plasma', 'coolwarm', 'inferno', etc.

    # Generate a list of colors from the colormap
    colors = [cmap(norm(value)) for value in values]

    # Create a visualization of the metrics
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(avg_metrics.keys()), y=values, palette= colors)
    plt.title("Evaluation Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("evaluation_metrics.png")
    plt.close()
    
    
    return avg_metrics, generated_texts

def interactive_evaluation(model, tokenizer, device, max_length=128):
    """Interactive evaluation where user can input formal text"""
    print("\n--- Interactive Evaluation ---")
    
    while True:
        user_input = input("\nEnter formal text (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        # Prepare input
        input_text = f"convert to informal: {user_input}"
        input_encoding = tokenizer(
            input_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,  # Add some randomness
                top_p=0.9
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Informal version: {generated_text}")

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer from the best checkpoint
    model_path = "model_checkpoints/best_model"  # Or use specific checkpoint
    try:
        print(f"Loading model from {model_path}...")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    except:
        print(f"Failed to load from {model_path}, using default model...")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    model.to(device)
    
    # Load test data
    from data_preparation import load_data
    file_path = "formal_informal_dataset.csv"  # Replace with your dataset path
    full_df = load_data(file_path)
    
    # Use a small portion for quick testing if the dataset is large
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(full_df, test_size=0.1, random_state=42)
    
    # Run evaluation
    print("Starting evaluation...")
    metrics, examples = evaluate_model(model, tokenizer, test_df, device)
    
    # Interactive evaluation
    interactive_evaluation(model, tokenizer, device)
