# BERTBERT2 Model Demo Instructions

This guide explains how to use the BERTBERT2 model through the BertBertScript.ipynb notebook.

## Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage Instructions

### 1. Training the Model

1. Open `BertBertScript.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells **up to the training part** to train the model
3. The training will create the necessary model files

### 2. Testing the Model

1. In the same notebook, run the **last cell**
2. This cell takes `all_data_demo.csv` as input
3. It will output a file with classification results

## Output

The model will classify text into one of four categories:
1. **relevant**: Genuine, helpful reviews about businesses
2. **rant**: Negative, emotional complaints  
3. **advertisement**: Promotional content or marketing messages
4. **spam**: Irrelevant, suspicious, or automated content

## Model Architecture

The BERTBERT2 model is a two-stage ensemble model:
- Uses DeBERTa-v3-small as the base model
- Classifies text into 4 categories: relevant, spam, advertisement, rant
- Trained on business review data

## Troubleshooting

- **CUDA not available**: The notebook will automatically fall back to CPU if CUDA is not available
- **Model loading errors**: Ensure all model files are in the correct paths
- **Memory issues**: Reduce batch size if you encounter memory problems

## Performance

- **CPU**: ~2-5 seconds per review
- **GPU (CUDA)**: ~0.1-0.5 seconds per review
- **Batch processing**: More efficient for large datasets
