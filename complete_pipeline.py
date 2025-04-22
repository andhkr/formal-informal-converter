import os
import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Adafactor,MT5Tokenizer, MT5ForConditionalGeneration

# Import functions from our scripts
from data_preparation import prepare_data
from model_training import train_model, evaluate
from model_evaluation import evaluate_model, interactive_evaluation

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.language == "hindi":
        # Load model and tokenizer
        if args.mode == "train" or args.mode == "full":
            print(f"Loading {args.model_name} model and tokenizer...")
            tokenizer = MT5Tokenizer.from_pretrained(args.model_name)
            model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
        else:
            print(f"Loading model from {args.model_path}...")
            try:
                tokenizer = MT5Tokenizer.from_pretrained(args.model_path)
                model = MT5ForConditionalGeneration.from_pretrained(args.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Falling back to {args.model_name}...")
                tokenizer = MT5Tokenizer.from_pretrained(args.model_name)
                model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        # Load model and tokenizer
        if args.mode == "train" or args.mode == "full":
            print(f"Loading {args.model_name} model and tokenizer...")
            tokenizer = T5Tokenizer.from_pretrained(args.model_name)
            model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        else:
            print(f"Loading model from {args.model_path}...")
            try:
                tokenizer = T5Tokenizer.from_pretrained(args.model_path)
                model = T5ForConditionalGeneration.from_pretrained(args.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Falling back to {args.model_name}...")
                tokenizer = T5Tokenizer.from_pretrained(args.model_name)
                model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)
    
    # Prepare data
    print("Preparing datasets...")
    train_dataloader, val_dataloader, test_dataloader, test_df = prepare_data(
        args.data_path, tokenizer, batch_size=args.batch_size
    )
    
    # Training
    if args.mode == "train" or args.mode == "full":
        print("Starting training...")
        
        # Using Adafactor optimizer which is memory efficient for T5
        optimizer = Adafactor(
            model.parameters(),
            lr=args.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
        
        model, tokenizer = train_model(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            save_dir=args.save_dir,
            save_steps=args.save_steps
        )
        print("Training complete!")
    
    # Evaluation
    if args.mode == "evaluate" or args.mode == "full":
        print("Starting evaluation...")
        metrics, examples = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            test_df=test_df,
            device=device,
            num_examples=args.num_examples
        )
    
    # Interactive mode
    if args.mode == "interactive":
        interactive_evaluation(model, tokenizer, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formal to Informal Text Conversion Pipeline")
    
    parser.add_argument("--language", type=str, default="english",
                        help="Name of language on Train model")
    parser.add_argument("--mode", type=str, default="full", choices=["train", "evaluate", "interactive", "full"],
                        help="Pipeline mode: train, evaluate, interactive, or full")
    parser.add_argument("--data_path", type=str, default="formal_informal_dataset.csv",
                        help="Path to dataset CSV file")
    parser.add_argument("--model_name", type=str, default="t5-small",
                        help="Name of base model to use")
    parser.add_argument("--model_path", type=str, default="model_checkpoints/best_model",
                        help="Path to load model from for evaluation")
    parser.add_argument("--save_dir", type=str, default="model_checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save model every X steps")
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Number of examples to show during evaluation")
    
    args = parser.parse_args()
    main(args)
