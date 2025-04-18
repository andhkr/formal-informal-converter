import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

# Load your dataset - adjust the path and format as needed
# Format assumption: CSV with 'formal' and 'informal' columns
def load_data(file_path):
    try:
        # For CSV format
        df = pd.read_csv(file_path)
        return df
    except:
        try:
            # For Excel format
            df = pd.read_excel(file_path)
            return df
        except:
            # For TSV format
            df = pd.read_csv(file_path, sep='\t')
            return df

# Custom dataset class
class FormalInformalDataset(Dataset):
    def __init__(self, formal_texts, informal_texts, tokenizer, max_length=128):
        self.formal_texts = formal_texts
        self.informal_texts = informal_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.formal_texts)

    def __getitem__(self, idx):
        formal_text = str(self.formal_texts[idx])
        informal_text = str(self.informal_texts[idx])
        
        # For T5, we prefix the input with a task descriptor
        input_text = f"convert to informal: {formal_text}"
        
        # Tokenize inputs and targets
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            informal_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by tokenizer
        input_ids = input_encoding.input_ids.squeeze()
        attention_mask = input_encoding.attention_mask.squeeze()
        labels = target_encoding.input_ids.squeeze()
        # Replace padding token id with -100 so it's ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def prepare_data(file_path, tokenizer, batch_size=16):
    print("Loading dataset...")
    df = load_data(file_path)
    
    # Print sample data
    print("\nData sample:")
    print(df.head())
    print(f"\nDataset size: {len(df)} pairs")
    
    # Basic data cleaning
    df['formal'] = df['formal'].str.strip()
    df['informal'] = df['informal'].str.strip()
    
    # Remove rows with missing values
    df = df.dropna()
    print(f"Dataset size after cleaning: {len(df)} pairs")
    
    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create datasets
    train_dataset = FormalInformalDataset(
        train_df['formal'].tolist(),
        train_df['informal'].tolist(),
        tokenizer
    )
    
    val_dataset = FormalInformalDataset(
        val_df['formal'].tolist(),
        val_df['informal'].tolist(),
        tokenizer
    )
    
    test_dataset = FormalInformalDataset(
        test_df['formal'].tolist(),
        test_df['informal'].tolist(),
        tokenizer
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    return train_dataloader, val_dataloader, test_dataloader, test_df

if __name__ == "__main__":
    # Test the data loading
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # Replace with your dataset path
    file_path = "formal_informal_dataset.csv"
    
    train_dataloader, val_dataloader, test_dataloader, test_df = prepare_data(
        file_path, tokenizer
    )
    
    # Check a sample batch
    for batch in train_dataloader:
        print("Input shape:", batch["input_ids"].shape)
        print("Attention mask shape:", batch["attention_mask"].shape)
        print("Labels shape:", batch["labels"].shape)
        
        # Decode a sample to verify
        sample_idx = 0
        input_text = tokenizer.decode(batch["input_ids"][sample_idx], skip_special_tokens=True)
        target_text = tokenizer.decode(
            [id if id != -100 else tokenizer.pad_token_id for id in batch["labels"][sample_idx]], 
            skip_special_tokens=True
        )
        print(f"\nSample input: {input_text}")
        print(f"Sample target: {target_text}")
        break
