# TikTok TechJam 2025 - BERT-Based Review Classification Model

## Overview
This project implements a sophisticated multi-tower BERT-based model for classifying restaurant reviews into four categories: `relevant`, `spam`, `rant`, and `advertisement`. The model combines text embeddings with engineered features and uses advanced techniques like PCA dimensionality reduction and multi-head attention.

## Quick Start

### Prerequisites

This project is meant to be run on Google Colab. However, you may also choose to run it on your local device. 

```bash
pip install torch transformers pandas numpy scikit-learn tqdm vaderSentiment textstat
```

### Running the Model

1. **Data Preparation** (if needed):
   ```python
   # Run the data preparation script first
   python prepare_data.py
   ```

2. **Training the Model**:
   ```python
   # Open BertBert2.ipynb in Jupyter/Colab
   # Run all cells to train the model
   # The model will be saved as 'bertbert2.bin'
   ```

3. **Testing the Model**:
   ```python
   # Load the trained model
   model = MainModel(...)
   model.load_state_dict(torch.load('bertbert2.bin'))
   model.eval()
   
   # Make predictions
   run bertberttest.ipynb

## 🔑 Key Features

### Feature Engineering (20 Features)
The model uses extensively engineered features across multiple domains:

#### Text-Based Features (8):
- **Sentiment Analysis**: VADER compound sentiment scores
- **Text Statistics**: Caps ratio, readability grade, word count, unique word ratio
- **Marketing Indicators**: Call-to-action detection, promotional offer detection
- **Punctuation**: Exclamation mark count

#### User Behavioral Features (4):
- **User Rating Patterns**: Average rating, rating consistency, review frequency
- **User Type Classification**: Binary flag for users who only give 5-star ratings

#### Business Context Features (3):
- **Business Metrics**: Average rating, total review count, picture count

#### Temporal Features (5):
- **Time Patterns**: Hour of day, day of week, time between reviews
- **Response Dynamics**: Business response delay
- **Content Alignment**: Category keyword matching

### Advanced Techniques

#### PCA Dimensionality Reduction
- **BERT Output**: 768 → 128 dimensions
- **Explained Variance**: 85.43% preserved
- **Purpose**: Reduces computational overhead while maintaining semantic information

#### Multi-Head Attention
- **Heads**: 8 attention heads
- **Query**: Text embeddings
- **Key/Value**: User + Business feature concatenation
- **Purpose**: Dynamic feature interaction and relevance scoring

#### Optimization Strategies
- **Focal Loss**: Handles class imbalance with α=-1, γ=2
- **Class Weights**: Logarithmic weighting based on class frequencies
- **Layer Unfreezing**: Top 2 BERT layers fine-tuned
- **Learning Rate**: 2e-5 with linear warmup scheduling

## 🏗️ Model Architecture

### Multi-Tower Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text Tower    │    │   User Tower    │    │ Business Tower  │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │    BERT     │ │    │ │  Embedding  │ │    │ │  Embedding  │ │
│ │ (base-uncased)│    │ │   (32 dim)  │ │    │ |  (64 dim)   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│         │       │    │         │       │    │         │       │
│         ▼       │    │         ▼       │    │         ▼       │
│ ┌─────────────┐ │    │                 │    │ ┌─────────────┐ │
│ │     PCA     │ │    │                 │    │ │     MLP     │ │
│ │ (768→128)   │ │    │                 │    │ │  (128→64)   │ │
│ └─────────────┘ │    │                 │    │ └─────────────┘ │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   Multi-Head Attention    │
                    │      (8 heads, 128 dim)   │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Classifier MLP       │
                    │   (256 → 128 → 4 classes) │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Output Classes       │
                    │  [relevant, spam, rant,  │
                    │   advertisement]          │
                    └───────────────────────────┘
```

### Tower Details

#### 1. Text Tower
- **Base Model**: BERT-base-uncased (768 dimensions)
- **PCA Reduction**: 768 → 128 dimensions (85.43% variance preserved)
- **Fine-tuning**: Top 2 layers unfrozen during training

#### 2. User Tower
- **Embedding Dimension**: 32
- **Features**: User ID encoding
- **Purpose**: Capture user-specific patterns and behaviors

#### 3. Business Tower
- **Components**:
  - Google Maps ID embedding (64 dim)
  - Price tier embedding (4 dim)
  - Category tags embedding bag (16 dim)
  - Misc tags embedding bag (16 dim)
  - Numerical features (20 dim)
- **MLP**: 128 → 64 dimensions
- **Purpose**: Business context and metadata representation

### Attention Mechanism
- **Query**: Text embeddings (128 dim)
- **Key/Value**: Concatenated user + business features (96 dim)
- **Heads**: 8 attention heads
- **Output**: Attended context vector (128 dim)

### Final Classification
- **Input**: Concatenated text + attended features (256 dim)
- **Hidden Layer**: 256 → 128 with ReLU and Dropout(0.5)
- **Output**: 128 → 4 classes (softmax)

## Model Performance

### Training Configuration
- **Batch Size**: 16
- **Epochs**: 15 (with early stopping)
- **Patience**: 3 epochs
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Loss Function**: Focal Loss with class weighting

### Validation Results
- **Accuracy**: 91.75%
- **Cohen's Kappa**: 0.7187
- **Class Performance**:
  - Relevant: 95.28% F1-score
  - Spam: 82.83% F1-score
  - Rant: 24.28% F1-score
  - Advertisement: 0.00% F1-score (current limitation)

## 🎯 Key Innovations

1. **Multi-Tower Architecture**: Separate processing for text, user, and business features
2. **PCA Integration**: Efficient BERT output compression
3. **Temporal Features**: Time-based behavioral patterns
4. **Attention Mechanism**: Dynamic feature interaction
5. **Focal Loss**: Better handling of class imbalance
6. **Layer Unfreezing**: Strategic BERT fine-tuning

## 📁 File Structure

```
├── BertBert2.ipynb          # Main training notebook
├── bertbert2.bin            # Trained model weights
├── prepare_data.py          # Data preprocessing script
├── training-data/           # Training datasets
└── my_trained_models/       # Additional model files
```

## Customization

### Adding New Features
1. Add feature calculation in the data preparation step
2. Update `NUM_NUMERICAL_FEATURES` constant
3. Modify the `ReviewDataset.__getitem__` method
4. Update the `BusinessTower` MLP input dimension

### Model Architecture Changes
1. Modify tower dimensions in the configuration section
2. Adjust attention mechanism parameters
3. Update classifier MLP architecture
4. Modify PCA dimensions if needed

## 📝 Citation

If you use this model in your research, please cite:
```
@misc{tiktok_techjam_2025_bert_classifier,
  title={Multi-Tower BERT with Feature Engineering for Review Classification},
  author={Your Name},
  year={2025},
  note={TikTok TechJam 2025 Submission}
}
```

## Contributing

This project was developed for the TikTok TechJam 2025 competition. For questions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
