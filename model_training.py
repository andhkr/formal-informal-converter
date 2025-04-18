import os
import time
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, Adafactor
from tqdm import tqdm
from data_preparation import prepare_data

def train_model(
    train_dataloader,
    val_dataloader,
    model,
    tokenizer,
    optimizer,
    device,
    num_epochs=5,
    save_dir="model_checkpoints",
    save_steps=500
):
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Save checkpoint
            if global_step % save_steps == 0:
                model_save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)
                print(f"\nCheckpoint saved to {model_save_path}")
                
                # Run validation
                val_loss = evaluate(model, val_dataloader, device)
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(save_dir, "best_model")
                    model.save_pretrained(best_model_path)
                    tokenizer.save_pretrained(best_model_path)
                    print(f"New best model saved to {best_model_path}")
                
                model.train()  # Switch back to train mode
        
        # End of epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch+1} complete. Average train loss: {avg_train_loss:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")
        
        # Validate after each epoch
        val_loss = evaluate(model, val_dataloader, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, "best_model")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"New best model saved to {best_model_path}")
    
    return model, tokenizer

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_name = "t5-small"  # Use t5-base for better results if resources allow
    print(f"Loading {model_name} model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    
    # Using Adafactor optimizer which is memory efficient for T5
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-4,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )
    
    # Prepare data
    print("Preparing datasets...")
    file_path = "formal_informal_dataset.csv"  # Replace with your dataset path
    train_dataloader, val_dataloader, test_dataloader, test_df = prepare_data(
        file_path, tokenizer, batch_size=8  # Use smaller batch size if memory is limited
    )
    
    # Train model
    print("Starting training...")
    num_epochs = 3  # Adjust based on your time constraints
    model, tokenizer = train_model(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        save_steps=100  # Save more frequently for a short training run
    )
    
    print("Training complete!")
