# BERTBERT: Metadata-Enhanced Review Classification

## Overview
This project implements a sophisticated multi-tower BERT-based model for classifying restaurant reviews into four categories: `relevant`, `spam`, `rant`, and `advertisement`. The model combines text embeddings with engineered features and uses advanced techniques like PCA dimensionality reduction and multi-head attention.

## Quick Start

### Prerequisites
This project is meant to be run on Google Colab. However, you may also choose to run it on your local device.

```bash
pip install torch transformers pandas numpy scikit-learn tqdm vaderSentiment textstat
```

### 1. Training the Model

1. Open `BertBertScript.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells **up to the training part** to train the model
3. The training will create the necessary model files

### 2. Testing the Model

1. In the same notebook, run the **last cell**
2. This cell takes `all_data_demo.csv` as input
3. It will output a file with classification results.

## Inspiration
The challenge we tackled was designing an AI system that outperforms current models in detecting the quality and relevancy of location-based reviews. Our inspiration came from the growing difficulty in identifying genuine reviews on platforms such as Google Maps and Yelp. Spam, misleading rants, and advertisements frequently distort the credibility of these platforms, which harms both users and businesses. We wanted to build a system that could accurately classify reviews as relevant, spam, rant, or advertisement, thereby ensuring more reliable insights for all stakeholders.

## What it does
Our solution integrates text-based analysis with metadata-driven feature engineering, moving beyond models that rely solely on text. The system classifies reviews into four categories:
- **Relevant**: Reviews that are related to the location.
- **Spam**: Reviews lacking sufficient relevant information about the location.
  - Example violation: "Great place!."
- **Rant**: Complaints or negative feedback not necessarily tied to actual visitors.
  - Example violation: "Never been here, but I heard it's terrible."
- **Advertisement**: Reviews that contain promotional material, links, or mention other establishments.
  - Example violation: "Best pizza! Visit www.pizzapromo.com for discounts!"

By combining embeddings from text with metadata features such as reviewer behaviour and posting time, the system provides more accurate detection of spam, rants, and advertisements.

## How we built it

### Data Collection and Cleaning
We collected reviews from Google Reviews (US market), the Yelp Open Dataset, and self-scraped Google Maps reviews in compliance with Google's Terms of Service. The cleaning process involved removing duplicates, filtering out non-English characters, and discarding truncated reviews that ended with ellipsis. We then manually labelled self-scraped reviews using strict classification policies to ensure consistency. In total, the dataset consisted of around 100,000 reviews for training and 4,000 manually labelled reviews for initial testing. Each data point contained the author's name, rating, text, and classification label.

**Data Processing Pipeline**: The raw data was processed using two key scripts:
- **`data_connect.py`**: Connects and merges data from multiple sources (Google Reviews, Yelp dataset, and self-scraped data) into a unified dataset
- **`process_merged_data.py`**: Processes the merged data by applying cleaning operations, feature engineering, and preparing the final dataset for model training

These scripts were essential in transforming raw, unprocessed review data into the clean, structured format required for effective model training.

### Model Development
We used supervised learning with our labelled data and tested different approaches. Baseline experiments were carried out with standard, more mainstream Large Language Models (LLMs) like Qwen3 8b and Gemini3 12b, pre-trained transformer models such as BERT (Bidirectional Encoder Representations from Transformers), and its variants distilBERT and roBERTa. We then built and experimented with an ensemble of different pre-trained models and ultimately ended up with our final solution of using an ensemble of 4 different models.

Our architecture employs a sophisticated multi-tower design:

### 🏗️ Multi-Tower Architecture

1. **Text Tower**: Utilizes a pre-trained variant of BERT, called bert-base-uncased. This tower converts the "review text" and "response text" features into a holistic 768-dimensional vector to capture semantic meaning and context, which is then compressed to 128 dimensions using Principal Component Analysis (PCA) while preserving 85.43% of the variance.

2. **User Tower**: Uses the embedding module from PyTorch to learn a unique 32-dimensional vector representation for each user, capturing user-specific behavioral patterns and review history.

3. **Business Tower**: Implements a Sequential Neural Network that consolidates all business-related features, including Google Maps ID embeddings (64 dimensions), price tier embeddings (4 dimensions), category tag embeddings (16 dimensions), misc tag embeddings (16 dimensions), and 20 engineered numerical features.

4. **Cross-Attention Mechanism**: The text tower queries the user tower and business tower outputs using multi-head attention (8 heads), allowing the model to dynamically weigh the importance of user behaviour and business context based on the specific language used in a review.

### 🔄 Data Flow Diagram

```
Raw Review Data
       ↓
┌─────────────────────────────────────────────────────────────┐
│                    Feature Engineering                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Text-Based  │  │User Behavior│  │  Business Context   │ │
│  │  Features   │  │  Features   │  │     Features        │ │
│  │   (8)      │  │    (4)      │  │       (3)           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Tower Model                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Text Tower │  │ User Tower  │  │   Business Tower    │ │
│  │ (BERT+PCA)  │  │  (32-dim)   │  │   (SNN + Embed)    │ │
│  │  (128-dim)  │  │             │  │     (120-dim)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│           ↓              ↓              ↓                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Cross-Attention Mechanism                  │ │
│  │              (Multi-Head: 8 heads)                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│                    Classification Layer                     │
│              (4 classes: Relevant, Spam, Rant, Ad)        │
└─────────────────────────────────────────────────────────────┘
```

1. **Text Tower**: Utilizes a pre-trained variant of BERT, called bert-base-uncased. This tower converts the "review text" and "response text" features into a holistic 768-dimensional vector to capture semantic meaning and context, which is then compressed to 128 dimensions using Principal Component Analysis (PCA) while preserving 85.43% of the variance.

2. **User Tower**: Uses the embedding module from PyTorch to learn a unique 32-dimensional vector representation for each user, capturing user-specific behavioral patterns and review history.

3. **Business Tower**: Implements a Sequential Neural Network that consolidates all business-related features, including Google Maps ID embeddings (64 dimensions), price tier embeddings (4 dimensions), category tag embeddings (16 dimensions), misc tag embeddings (16 dimensions), and 20 engineered numerical features.

4. **Cross-Attention Mechanism**: The text tower queries the user tower and business tower outputs using multi-head attention (8 heads), allowing the model to dynamically weigh the importance of user behaviour and business context based on the specific language used in a review.

The model's output is then passed onto a classifier to make the final prediction. To prevent overfitting, we experimented with applied techniques such as dropout, regularisation, normalisation, and early stopping. Iterating on larger datasets and more advanced models helped to improve generalisation.

### Feature Engineering
Unlike most models that rely only on text, we engineered 20 comprehensive metadata-based features across multiple domains to provide additional context:

#### Text-Based Features (8 features):
- **Sentiment Analysis**: VADER compound sentiment scores for emotional tone assessment
- **Text Statistics**: Caps ratio (uppercase to alphabetic character ratio), Flesch-Kincaid readability grade level, word count, and unique word ratio
- **Marketing Indicators**: Binary flags for call-to-action detection (visit, check out, try our, don't miss, shop now) and promotional offer detection (deal, offer, discount, promotion, free, sale)
- **Punctuation Analysis**: Exclamation mark count as engagement indicator

#### User Behavioral Features (4 features):
- **User Rating Patterns**: Average rating across all reviews, rating consistency (standard deviation), and review frequency
- **User Type Classification**: Binary flag identifying users who exclusively give 5-star ratings, indicating potential review farming behavior

#### Business Context Features (3 features):
- **Business Metrics**: Average rating, total review count, and picture count associated with the business

#### Temporal Features (5 features):
- **Time Patterns**: Hour of day (0-23) and day of week (0-6) when reviews were posted
- **Response Dynamics**: Time difference between review submission and business response in hours
- **Content Alignment**: Count of category tags that appear in the review text, measuring topical relevance

All numerical features are normalized using StandardScaler to ensure consistent model training and prevent feature dominance.

### Advanced Optimization Techniques
Our model incorporates several sophisticated optimization strategies:

- **Focal Loss**: Implements focal loss with α=-1 and γ=2 parameters to better handle class imbalance, particularly for the underrepresented spam, rant, and advertisement classes
- **Class Weighting**: Applies logarithmic weighting based on class frequencies to address dataset skewness
- **Strategic Layer Unfreezing**: Selectively unfreezes the top 2 BERT layers during fine-tuning to balance computational efficiency with model adaptability
- **Learning Rate Scheduling**: Employs AdamW optimizer with a learning rate of 2e-5 and linear warmup scheduling for stable training

## Evaluation and Iteration
We benchmarked BERTBERT against OpenAI ChatGPT and other baseline models. The inclusion of metadata provided measurable improvements in detecting spam, advertisements, and rants. Evaluation metrics included precision, recall, F1-score, accuracy, and Cohen's Kappa, which together gave a more reliable picture of model performance.

Our final model achieved:
- **Overall Accuracy**: 91.75%
- **Cohen's Kappa**: 0.7187
- **Class Performance**:
  - Relevant: 95.28% F1-score
  - Spam: 82.83% F1-score
  - Rant: 24.28% F1-score
  - Advertisement: 0.00% F1-score (current limitation)

## Challenges we ran into
One of the challenges we faced was data quality. Initially, we followed the instructions given and tried labelling the given dataset with ChatGPT-4o. However, we realised that the LLM's output was unreliable in labelling the data. As such, we looked to fine-tune our own labelling model on a small manually labelled dataset. We fine-tuned our model which uses DebertaV2forSequenceClassification and then used the model to label a much larger dataset of about 1.5 million instances of data.

However, while trying to train our "labelling model", we faced a bunch of problems due to the very skewed dataset. In our initially manually labelled dataset of about 1.5k instances, we had less than 50 combined instances of "spam", "advertisement" and "rant". This led to the model being unable to generalize well to these classes. We tried a few solutions to solve this:

**SMOTE (Synthetic Minority Oversampling Technique)**: We considered this, but then we had concerns with the introduction of synthetic data introducing too much noise. Our experimentation with it proved us right as we constantly got low f1 scores with this method.

**Unequal weights according to the frequency of the data**: We use this in some of our models, but the tweaking of the weights have also been a challenge as we had to spend a large amount of time to find appropriate weights.

**Supplementing the dataset with more instances of "spam", "advertisement" and "rant"**: This was what we ultimately turned to, as we felt that the google reviews dataset had too little "advertisement", "spam" and "rant" instances (which makes sense given how Google likely already has a filtering algorithm in place). As such, we looked to alternative datasets, landing on Yelp reviews and supplementing our data with it.

Quality of data was also an issue, many reviews ended with ellipsis, which meant that the content was truncated. Reviews also had empty fields of data and we had to carefully handle such cases during preprocessing. Pandas came particularly helpful in this aspect as we managed to clean the data quickly and carefully.

Computational resources were also a huge challenge. As none of our team had access to GPUs other than the free runtime on Google Colab, we were unable to train our models on very large datasets. Actually, we had prepared and cleaned a dataset of 1.5million instances of data with our labeling model. However, due to lack of time and GPU access, we had to settle for using a much smaller dataset of about 50 thousand instances to train our final model on. We also used dimensionality reduction methods such as Principal Component Analysis (PCA) to decrease the computational load, even though we would lose some of the signal. We thought that this was worth it as we would be able to increase the rate at which we test new models, while the signal lost was relatively very little.

Finally, the metadata sometimes introduced noise, as some behavioural signals produced spurious correlations. Normalisation and careful validation were required to reduce such effects. We approached the metadata in this manner, coming up with theories on what we think would create a signal, and testing it. We did not want to approach the data in a datamining manner, where we looked for correlation between different features as there is a higher chance that it is a coincidental relation idiosyncratic to this dataset.

## Accomplishments
We are proud of several key accomplishments.

First, we were able to train a labeling model that beat out OpenAI's GPT-4o in the labeling of data.

Second, that we were able to work together as a team to engineer solutions to the many problems that we have faced, especially for maintaining data quality.

Third, that we were able to navigate the lack of computational resources to still build a model in which the metrics are respectable.

Lastly, that we were able to learn about and implement machine learning concepts which most of us have never used before.

## What we learned
From this project, we have learned that real life machine learning applications stem much further than just experimentation with models. In real applications it is very hard to find a clean and readily labeled dataset as we see with Kaggle competitions. We have realised that most of the 80% problems are data-related, and that good quality data collection is what makes a good model.

## What's next for BERTBERT
Looking forward, we plan to first solve our problem of the lack of GPUs and computation power. This would allow us to be able to realistically scale up the training to train on all of the originally planned training set of 1.5 million data instances. We would also aim to get a larger manually labelled dataset through engaging paid manual labor to label the data, on which we would then train our labeling model, so that we could get better pseudo-labeled data to train our final model on. We could also explore the possibility of getting a data labelling vendor such as Scale AI to handle the data so we would have the best quality of data possible. We would then be able to fine-tune our models better and even experiment with different model architectures. We also want to integrate explainability tools such as SHAP or LIME so that users and businesses can understand why a review was flagged. Another goal is to build a real-time API that businesses can use to filter reviews as they are posted. In addition, we aim to extend the model to other domains such as Amazon or TripAdvisor reviews. Finally, with experimentation, we will continue to refine the optimisation process, particularly loss functions, to further improve handling of class imbalance.

## 📁 File Structure

```
├── BertBertScript.ipynb     # Main training notebook
├── data-processing-scripts/  # Data processing and merging scripts
│   ├── data_connect.py      # Script for connecting and merging raw data sources
│   └── process_merged_data.py # Script for processing merged data for model training
├── training-data/            # Training datasets
├── my_trained_models/        # Trained model files
└── all_data_demo.csv         # Demo data for testing
```

## Training Results Summary

### Model A: Feature Engineering (No PCA)
**Setup**
- Device: cuda
- Samples: Train = 46,031 | Validation = 11,508
- Unfrozen top 2 BERT layers
- Class Weights: [0.0455, 0.6028, 1.4375, 1.9142]
- Early Stopping patience = 3

**Key Metrics by Epoch**
| Epoch | Train Loss | Val Loss | Kappa | Accuracy | Notes |
|-------|------------|----------|-------|----------|-------|
| 1     | 0.0070     | 0.0049   | 0.654 | 0.889   | Saved (best) |
| 2     | 0.0047     | 0.0041   | 0.701 | 0.909   | Saved (best) |
| 3     | 0.0037     | 0.0049   | 0.795 | 0.944   | No improve |
| 4     | 0.0029     | 0.0040   | 0.742 | 0.924   | Saved (best) |
| 5     | 0.0024     | 0.0045   | 0.749 | 0.927   | No improve |
| 6     | 0.0020     | 0.0067   | 0.803 | 0.947   | No improve |
| 7     | 0.0016     | 0.0062   | 0.792 | 0.942   | Early stop |

**Best Performance**
- Validation Accuracy: 94.7%
- Cohen's Kappa: 0.803
- Strong performance on Relevant + Spam
- Weakest class: Advertisement (precision < 0.10)

### Model B: Feature Engineering + PCA
**Setup**
- Device: cuda
- Samples: Train = 46,031 | Validation = 11,508
- PCA variance retained: 85.4%
- Unfrozen top 2 BERT layers
- Class Weights: [0.0455, 0.6028, 1.4375, 1.9142]
- Early Stopping patience = 3

**Key Metrics by Epoch**
| Epoch | Train Loss | Val Loss | Kappa | Accuracy | Notes |
|-------|------------|----------|-------|----------|-------|
| 1     | 0.0071     | 0.0051   | 0.719 | 0.918   | Saved (best) |
| 2     | 0.0047     | 0.0044   | 0.677 | 0.899   | Saved (best) |
| 3     | 0.0038     | 0.0042   | 0.739 | 0.924   | Saved (best) |
| 4     | 0.0032     | 0.0046   | 0.678 | 0.898   | No improve |
| 5     | 0.0026     | 0.0049   | 0.742 | 0.924   | No improve |
| 6     | 0.0022     | 0.0055   | 0.781 | 0.939   | Early stop |

**Best Performance**
- Validation Accuracy: 93.9%
- Cohen's Kappa: 0.781
- Better recall for minority classes (esp. Advertisement)
- Still weaker precision on Advertisement/Rant

### Comparison
| Metric | Model A (Feat Eng) | Model B (Feat Eng + PCA) |
|--------|-------------------|---------------------------|
| Best Accuracy | 94.7% | 93.9% |
| Best Cohen's Kappa | 0.803 | 0.781 |
| Minority Class Recall | Lower | Higher (esp. Ads) |
| Stability across epochs | More stable | Slightly more volatile |

✅ **Takeaway:**
- Model A achieves higher raw accuracy and agreement (Kappa).
- Model B sacrifices some accuracy but slightly improves recall on rare classes.
- Choose Model A if prioritizing overall performance; choose Model B if focusing on minority class fairness.
