# Training Results Summary

## Model A: Feature Engineering (No PCA)
**Source**: `BERTBERT1.ipynb`

### Setup
- **Device**: cuda
- **Samples**: Train = 46,031 | Validation = 11,508
- **Architecture**: Unfrozen top 2 BERT layers
- **Class Weights**: [0.0455, 0.6028, 1.4375, 1.9142]
- **Early Stopping patience**: 3

### Key Metrics by Epoch

| Epoch | Train Loss | Val Loss | Kappa | Accuracy | Notes |
|-------|------------|----------|-------|----------|-------|
| 1     | 0.0070     | 0.0049   | 0.654 | 0.889   | Saved (best) |
| 2     | 0.0047     | 0.0041   | 0.701 | 0.909   | Saved (best) |
| 3     | 0.0037     | 0.0049   | 0.795 | 0.944   | No improve |
| 4     | 0.0029     | 0.0040   | 0.742 | 0.924   | Saved (best) |
| 5     | 0.0024     | 0.0045   | 0.749 | 0.927   | No improve |
| 6     | 0.0020     | 0.0067   | 0.803 | 0.947   | No improve |
| 7     | 0.0016     | 0.0062   | 0.792 | 0.942   | Early stop |

### Best Performance
- **Validation Accuracy**: 94.7%
- **Cohen's Kappa**: 0.803
- **Strengths**: Strong performance on Relevant + Spam
- **Weaknesses**: Weakest class: Advertisement (precision < 0.10)

---

## Model B: Feature Engineering + PCA
**Source**: `BERTBERTScript.ipynb`

### Setup
- **Device**: cuda
- **Samples**: Train = 46,031 | Validation = 11,508
- **PCA variance retained**: 85.4%
- **Architecture**: Unfrozen top 2 BERT layers
- **Class Weights**: [0.0455, 0.6028, 1.4375, 1.9142]
- **Early Stopping patience**: 3

### Key Metrics by Epoch

| Epoch | Train Loss | Val Loss | Kappa | Accuracy | Notes |
|-------|------------|----------|-------|----------|-------|
| 1     | 0.0071     | 0.0051   | 0.719 | 0.918   | Saved (best) |
| 2     | 0.0047     | 0.0044   | 0.677 | 0.899   | Saved (best) |
| 3     | 0.0038     | 0.0042   | 0.739 | 0.924   | Saved (best) |
| 4     | 0.0032     | 0.0046   | 0.678 | 0.898   | No improve |
| 5     | 0.0026     | 0.0049   | 0.742 | 0.924   | No improve |
| 6     | 0.0022     | 0.0055   | 0.781 | 0.939   | Early stop |

### Best Performance
- **Validation Accuracy**: 93.9%
- **Cohen's Kappa**: 0.781
- **Strengths**: Better recall for minority classes (esp. Advertisement)
- **Weaknesses**: Still weaker precision on Advertisement/Rant

---

## Model Comparison

| Metric | Model A (Feat Eng) | Model B (Feat Eng + PCA) |
|--------|-------------------|--------------------------|
| **Best Accuracy** | 94.7% | 93.9% |
| **Best Cohen's Kappa** | 0.803 | 0.781 |
| **Minority Class Recall** | Lower | Higher (esp. Ads) |
| **Stability across epochs** | More stable | Slightly more volatile |

## ✅ Key Takeaways

- **Model A** achieves higher raw accuracy and agreement (Kappa).
- **Model B** sacrifices some accuracy but slightly improves recall on rare classes.
- **Choose Model A** if prioritizing overall performance.
- **Choose Model B** if focusing on minority class fairness.

---

*Generated from training runs:*
- *Model A: BERTBERT1.ipynb*
- *Model B: BERTBERTScript.ipynb*
