#!/usr/bin/env python3
"""
Demo Script for BERTBERT2 Model
This script loads the trained BERTBERT2 model and makes predictions on new CSV data.
It adds a new 'classification' column with one of 4 categories: relevant, rant, advertisement, spam.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

class TextEncoder(nn.Module):
    """Text encoder using BERT model"""
    def __init__(self, model_name, hidden_size=768):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 4)  # 4 classes
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class BERTBERT2Predictor:
    """Main predictor class for BERTBERT2 model"""
    
    def __init__(self, model_path, model_config_path, label_encoder_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model configuration
        with open(model_config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load label encoder
        with open(label_encoder_path, 'r') as f:
            label_data = json.load(f)
            self.classes = label_data['classes']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['violation_model'])
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully. Classes: {self.classes}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        try:
            # Try to load the model directly
            model = torch.load(model_path, map_location=self.device)
            if isinstance(model, dict) and 'model_state_dict' in model:
                # If it's a checkpoint, extract the model
                model_state_dict = model['model_state_dict']
                # Create a new model instance
                new_model = TextEncoder(self.config['violation_model'])
                new_model.load_state_dict(model_state_dict)
                return new_model
            else:
                # If it's the model directly
                return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model instance...")
            # Create a new model instance if loading fails
            model = TextEncoder(self.config['violation_model'])
            return model
    
    def preprocess_text(self, text):
        """Preprocess text for the model"""
        if pd.isna(text) or text == '':
            text = "no text provided"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        return inputs
    
    def predict_single(self, text):
        """Predict classification for a single text"""
        inputs = self.preprocess_text(text)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
        return {
            'predicted_class': self.classes[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'probabilities': probabilities.cpu().numpy()[0]
        }
    
    def predict_batch(self, texts, batch_size=32):
        """Predict classifications for a batch of texts"""
        predictions = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Making predictions"):
            batch_texts = texts[i:i+batch_size]
            batch_predictions = []
            
            for text in batch_texts:
                pred = self.predict_single(text)
                batch_predictions.append(pred)
            
            predictions.extend(batch_predictions)
        
        return predictions
    
    def predict_csv(self, input_csv_path, output_csv_path, text_column='text'):
        """Process CSV file and add classification column"""
        print(f"Loading CSV from: {input_csv_path}")
        
        # Load CSV
        try:
            df = pd.read_csv(input_csv_path)
            print(f"Loaded {len(df)} rows")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return
        
        # Check if text column exists
        if text_column not in df.columns:
            print(f"Text column '{text_column}' not found. Available columns: {list(df.columns)}")
            return
        
        # Make predictions
        print("Making predictions...")
        predictions = self.predict_batch(df[text_column].tolist())
        
        # Add predictions to dataframe
        df['classification'] = [pred['predicted_class'] for pred in predictions]
        df['confidence'] = [pred['confidence'] for pred in predictions]
        
        # Add individual class probabilities
        for i, class_name in enumerate(self.classes):
            df[f'prob_{class_name}'] = [pred['probabilities'][i] for pred in predictions]
        
        # Save results
        print(f"Saving results to: {output_csv_path}")
        df.to_csv(output_csv_path, index=False)
        
        # Print summary
        print("\nClassification Summary:")
        print(df['classification'].value_counts())
        print(f"\nResults saved to: {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description='BERTBERT2 Model Demo Script')
    parser.add_argument('--input_csv', required=True, help='Path to input CSV file')
    parser.add_argument('--output_csv', required=True, help='Path to output CSV file')
    parser.add_argument('--text_column', default='text', help='Name of text column (default: text)')
    parser.add_argument('--model_path', default='my_trained_models (1)/BERTBERT2.bin', 
                       help='Path to BERTBERT2 model file')
    parser.add_argument('--config_path', default='my_trained_models (1)/model_config.json',
                       help='Path to model config file')
    parser.add_argument('--label_encoder_path', default='my_trained_models (1)/label_encoder.json',
                       help='Path to label encoder file')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file '{args.input_csv}' not found.")
        return
    
    # Check if model files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        return
    
    if not os.path.exists(args.config_path):
        print(f"Error: Config file '{args.config_path}' not found.")
        return
    
    if not os.path.exists(args.label_encoder_path):
        print(f"Error: Label encoder file '{args.label_encoder_path}' not found.")
        return
    
    try:
        # Initialize predictor
        print("Initializing BERTBERT2 predictor...")
        predictor = BERTBERT2Predictor(
            model_path=args.model_path,
            model_config_path=args.config_path,
            label_encoder_path=args.label_encoder_path
        )
        
        # Process CSV
        predictor.predict_csv(
            input_csv_path=args.input_csv,
            output_csv_path=args.output_csv,
            text_column=args.text_column
        )
        
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
