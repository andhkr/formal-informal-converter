# Formal to Informal Text Conversion

This project implements a neural machine translation approach to convert formal English text to informal English. It uses a T5 transformer model fine-tuned on a parallel corpus of formal-informal text pairs.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Requirements](#dataset-requirements)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Pipeline Script](#pipeline-script)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Interactive Mode](#interactive-mode)
  - [Web Demo](#web-demo)
- [Command Line Arguments](#command-line-arguments)
- [Output Files](#output-files)
- [Customization](#customization)
- [Tips & Troubleshooting](#tips--troubleshooting)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd formal-to-informal-converter
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required NLTK resources:
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

## Project Structure

```
formal-to-informal-converter/
├── data_preparation.py             # Script for preparing dataset
├── model_training.py               # Script for model training
├── model_evaluation.py             # Script for evaluating model
├── complete_pipeline.py            # Main pipeline script
├── alternate_hindi_pipeline.ipynb  # Alternate Pipeline for Hindi
├── web_demo.py                     # Simple web interface for the model
├── templates/                      # Web templates directory
│   └── index.html                  # Web interface HTML
├── model_checkpoints/              # Directory for saved models
└── README.md                       # This file
```

## Dataset Requirements

Your dataset should be a CSV file with at least two columns:
- A column named "formal" containing the formal text examples
- A column named "informal" containing the corresponding informal versions

Example format:
```
formal,informal
"I would like to inquire about your services.","I wanna ask about your services."
"Please do not hesitate to contact me.","Feel free to hit me up."
```

## Quick Start

To run the complete pipeline (train, evaluate, and start interactive mode) with default settings:

```bash
python complete_pipeline.py --mode full --data_path your_dataset.csv
```

## Usage Guide

### Pipeline Script

The `complete_pipeline.py` script is the main entry point that ties together all components. It handles data preparation, model training, evaluation, and interactive testing.

#### Basic Usage

```bash
python complete_pipeline.py --mode <mode> --data_path <path_to_dataset> --language <language> 
```

where 

1. `language` can be:
- `hindi`
- `english`

2. `<mode>` can be:
- `train`: Only train the model
- `evaluate`: Only evaluate an existing model
- `interactive`: Start an interactive session to test the model
- `full`: Run the complete pipeline (data prep → training → evaluation)


### Training

To train a model from scratch:

```bash
python complete_pipeline.py --language hindi --mode train --data_path your_dataset.csv --num_epochs 5 --model_name google/mt5-small
```

Training will:
1. Load and prepare your dataset
2. Initialize a T5 model
3. Fine-tune it on your data
4. Save checkpoints periodically
5. Save the best model based on validation loss

### Evaluation

To evaluate a trained model:

```bash
python complete_pipeline.py --mode evaluate --data_path your_dataset.csv --model_path model_checkpoints/best_model
```

Evaluation will:
1. Load your trained model
2. Run predictions on the test set
3. Calculate metrics (BLEU, METEOR, ROUGE)
4. Display sample predictions
5. Save results and metrics to CSV files
6. Generate a visualization of metrics

### Interactive Mode

To interact with your model:

```bash
python complete_pipeline.py --mode interactive --model_path model_checkpoints/best_model
```

This will start a command-line interface where you can enter formal text and get the informal conversion in real-time.


## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Pipeline mode: train, evaluate, interactive, or full | full |
| `--data_path` | Path to dataset CSV file | formal_informal_dataset.csv |
| `--model_name` | Name of base model to use | t5-small |
| `--model_path` | Path to load model from for evaluation | model_checkpoints/best_model |
| `--save_dir` | Directory to save model checkpoints | model_checkpoints |
| `--batch_size` | Batch size for training and evaluation | 8 |
| `--num_epochs` | Number of training epochs | 3 |
| `--learning_rate` | Learning rate for optimizer | 1e-4 |
| `--save_steps` | Save model every X steps | 100 |
| `--num_examples` | Number of examples to show during evaluation | 5 |
|`--language` | Name of Language on Which you train model | english |

## Output Files

During training and evaluation, the following files will be generated:

- `model_checkpoints/checkpoint-{step}/`: Checkpoint saved during training
- `model_checkpoints/best_model/`: Best model based on validation loss
- `generation_results.csv`: CSV file with formal, reference informal, and generated informal texts
- `evaluation_metrics.csv`: CSV file with evaluation metrics
- `evaluation_metrics.png`: Visualization of evaluation metrics

## Customization

### Using Different Models

You can use different T5 model sizes by changing the `--model_name` parameter:
- `t5-small` (60M parameters): Fast but less accurate
- `t5-base` (220M parameters): Good balance of speed and quality
- `t5-large` (770M parameters): High quality but requires more resources

Example:
```bash
python complete_pipeline.py --mode train --data_path your_dataset.csv --model_name t5-base
```

### Memory Optimization

If you encounter out-of-memory errors:
1. Reduce the batch size: `--batch_size 4` or even `--batch_size 2`
2. Use a smaller model: `--model_name t5-small`
3. Reduce the maximum sequence length (requires code modification)

## Tips & Troubleshooting

### Best Practices
- Start with a small number of epochs (2-3) to get quick results
- For better quality, increase epochs to 5-10 if you have the time and resources
- Save checkpoints frequently during initial experiments (`--save_steps 50`)
- Use a GPU for significantly faster training

### Common Issues
- **Out of memory errors**: Reduce batch size or use a smaller model
- **Poor conversion quality**: Try using a larger model (t5-base instead of t5-small)
- **Training too slow**: Ensure you're using a GPU, or reduce dataset size for testing
- **Model generates same output**: Check your dataset for quality and diversity

### Example Training Times
- t5-small with 5000 examples: ~30 minutes on GPU, several hours on CPU
- t5-base with 5000 examples: ~1-2 hours on GPU, not recommended on CPU

### Performance Metrics
Typical performance ranges on a good dataset:
- BLEU-1: 0.4-0.7
- ROUGE-L: 0.5-0.8
- METEOR: 0.3-0.6

Lower scores may indicate issues with your dataset or training process.
