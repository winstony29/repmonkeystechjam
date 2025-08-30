# BERTBERT2 Model Demo Script

This demo script showcases the BERTBERT2 model in action, classifying text into one of four categories: **relevant**, **rant**, **advertisement**, or **spam**.

## Features

- Loads the pre-trained BERTBERT2 model
- Processes CSV files with text data
- Adds classification predictions with confidence scores
- Supports batch processing for efficiency
- Provides individual class probabilities

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python demo_script.py --input_csv your_data.csv --output_csv results.csv
```

### Advanced Usage

```bash
python demo_script.py \
    --input_csv your_data.csv \
    --output_csv results.csv \
    --text_column review_text \
    --model_path path/to/BERTBERT2.bin \
    --config_path path/to/model_config.json \
    --label_encoder_path path/to/label_encoder.json
```

### Parameters

- `--input_csv`: Path to input CSV file (required)
- `--output_csv`: Path to output CSV file (required)
- `--text_column`: Name of the text column (default: "text")
- `--model_path`: Path to BERTBERT2.bin model file (default: "my_trained_models (1)/BERTBERT2.bin")
- `--config_path`: Path to model config file (default: "my_trained_models (1)/model_config.json")
- `--label_encoder_path`: Path to label encoder file (default: "my_trained_models (1)/label_encoder.json")

## Example

### Input CSV Format

Your CSV should have a text column (default name: "text") containing the reviews/text to classify:

```csv
user_id,author_name,business_name,rating,text,review_time
1,John Doe,Restaurant,5.0,"Amazing food and great service!",2024-01-15
2,Jane Smith,Store,1.0,"Terrible experience!",2024-01-16
```

### Output CSV Format

The script adds several new columns:

- `classification`: Predicted category (relevant, rant, advertisement, spam)
- `confidence`: Confidence score for the prediction
- `prob_relevant`: Probability for relevant class
- `prob_rant`: Probability for rant class
- `prob_advertisement`: Probability for advertisement class
- `prob_spam`: Probability for spam class

## Test the Script

1. Use the provided example data:
```bash
python demo_script.py --input_csv example_test_data.csv --output_csv test_results.csv
```

2. Check the results:
```bash
head -10 test_results.csv
```

## Model Architecture

The BERTBERT2 model is a two-stage ensemble model:
- Uses DeBERTa-v3-small as the base model
- Classifies text into 4 categories: relevant, spam, advertisement, rant
- Trained on business review data

## Troubleshooting

- **CUDA not available**: The script automatically falls back to CPU if CUDA is not available
- **Model loading errors**: Ensure all model files are in the correct paths
- **Memory issues**: Reduce batch size by modifying the `batch_size` parameter in the code

## Performance

- **CPU**: ~2-5 seconds per review
- **GPU (CUDA)**: ~0.1-0.5 seconds per review
- **Batch processing**: More efficient for large datasets

## Output Categories

1. **relevant**: Genuine, helpful reviews about businesses
2. **rant**: Negative, emotional complaints
3. **advertisement**: Promotional content or marketing messages
4. **spam**: Irrelevant, suspicious, or automated content
