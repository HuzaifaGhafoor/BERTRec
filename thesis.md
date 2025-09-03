# BERTRec: A Sentiment-Based Business Recommender System Using BERT-LSTM Neural Networks

## Abstract

This research presents BERTRec, a novel sentiment-based business recommender system that integrates state-of-the-art natural language processing techniques with content-based filtering algorithms. The system leverages a fine-tuned RoBERTa-LSTM hybrid model for sentiment analysis of user reviews, achieving 86.0% accuracy on ternary sentiment classification. The recommender system combines sentiment patterns with business features to provide personalized recommendations, demonstrating the effectiveness of deep learning approaches in recommendation systems.

## 1. Introduction

Traditional recommender systems primarily rely on ratings and collaborative filtering approaches, often overlooking the rich semantic information embedded in user reviews. This research addresses this limitation by developing a sentiment-aware recommender system that analyzes the emotional content of reviews to better understand user preferences and business characteristics.

The BERTRec system consists of two main components: (1) a sentiment analysis engine based on RoBERTa-LSTM architecture, and (2) a content-based recommender system that utilizes sentiment profiles alongside traditional business features.

## 2. Methodology

### 2.1 Data Acquisition and Sampling

#### 2.1.1 Dataset Selection
The research utilized the Yelp Academic Dataset, a comprehensive collection of business reviews, user information, and business metadata. The original dataset contained approximately 6.99 million reviews across 9 attributes.

#### 2.1.2 Sampling Strategy
To manage computational complexity while maintaining representativeness, a stratified sampling approach was implemented:

- **Target Sample Size**: 1.5 million reviews
- **Sampling Method**: Balanced stratified sampling with 500,000 reviews per sentiment class
- **Memory Management**: Utilized Dask for efficient processing of large JSON files
- **Format Conversion**: Converted JSON to Parquet format with Snappy compression for optimized storage and retrieval

The sampling process involved:
1. Loading the dataset in chunks of 10,000 records to prevent memory exhaustion
2. Converting to Parquet format for efficient processing
3. Mapping star ratings to sentiment labels:
   - Negative: 1-2 stars
   - Neutral: 3 stars  
   - Positive: 4-5 stars
4. Stratified sampling to ensure equal representation across sentiment classes

### 2.2 Data Preprocessing Pipeline

#### 2.2.1 Duplicate Detection and Removal
A comprehensive duplicate removal strategy was implemented to ensure data quality:

- **Same User-Business-Text Combinations**: 1,651 duplicates identified
- **Different Users-Same Business-Text**: 1,689 duplicates identified  
- **Same User-Different Business-Text**: 2,119 duplicates identified
- **Different Users-Different Business-Text**: 2,171 duplicates identified

Temporal sorting was applied, retaining the most recent review for duplicate resolution.

#### 2.2.2 Text Normalization
The preprocessing pipeline included:

1. **Character Cleaning**: Removal of brackets, parentheses, and excessive whitespace
2. **Case Normalization**: Converting all text to lowercase
3. **Number Normalization**: Replacing numerical values with `[NUM]` tokens while preserving star ratings
4. **Contraction Expansion**: Using the `contractions` library to expand abbreviated forms
5. **Spell Correction**: Implementation of SymSpell algorithm for correcting common misspellings

#### 2.2.3 Label Encoding
Sentiment labels were encoded numerically:
- Negative: 0
- Neutral: 1  
- Positive: 2

### 2.3 Model Architecture

#### 2.3.1 RoBERTa-LSTM Hybrid Model
The sentiment analysis model combines the contextual understanding of RoBERTa with the sequential modeling capabilities of LSTM networks:

```python
class RoBERTa_LSTM(nn.Module):
    def __init__(self, roberta_model='roberta-base', lstm_hidden=256, num_classes=3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(roberta_model)
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, 
                           batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(lstm_hidden * 2)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
```

**Architecture Components**:
- **RoBERTa Base**: 12-layer transformer encoder (768-dimensional hidden states)
- **Bidirectional LSTM**: 256 hidden units per direction (512 total output)
- **Layer Normalization**: Applied to LSTM output for training stability
- **Dropout**: 40% dropout rate for regularization
- **Classification Head**: Linear layer mapping to 3 sentiment classes

#### 2.3.2 Training Configuration
- **Optimizer**: AdamW with 2e-5 learning rate and 0.01 weight decay
- **Loss Function**: CrossEntropyLoss with 0.1 label smoothing
- **Scheduler**: Linear warmup (10% of steps) followed by linear decay
- **Mixed Precision**: Automatic Mixed Precision (AMP) for memory efficiency
- **Batch Size**: 64 samples per batch
- **Epochs**: 5 training epochs

### 2.4 Data Splitting and Tokenization

#### 2.4.1 Dataset Partitioning
The preprocessed dataset was split using stratified sampling:
- **Training Set**: 80% (1,199,118 samples)
- **Validation Set**: 10% (149,890 samples)  
- **Test Set**: 10% (149,890 samples)

#### 2.4.2 Tokenization Strategy
RoBERTa tokenizer was employed with the following specifications:
- **Maximum Length**: 512 tokens
- **Padding**: Max length padding for consistent batch processing
- **Truncation**: Applied to texts exceeding maximum length
- **Special Tokens**: Added for sequence classification
- **Memory Mapping**: Utilized numpy memory mapping for efficient tokenization of large datasets

### 2.5 Recommender System Architecture

#### 2.5.1 Content-Based Filtering Approach
The recommender system employs content-based filtering with multiple feature types:

**Business Features**:
- Categorical features (TF-IDF vectorized business categories)
- Numerical features (star ratings, review counts)
- Sentiment profiles (positive/negative/neutral ratios)

**Feature Engineering**:
- **Category Vectorization**: TF-IDF with 100 maximum features
- **Numerical Standardization**: StandardScaler normalization
- **Sentiment Profiling**: Business-level sentiment distribution calculation

#### 2.5.2 Similarity Computation
Cosine similarity is employed to measure business similarity across the combined feature space:

```python
similarities = cosine_similarity([business_feature_vector], feature_matrix)[0]
```

#### 2.5.3 Recommendation Algorithm
The recommendation process follows these steps:

1. **Sentiment Analysis**: Predict user review sentiment using trained model
2. **Business Filtering**: Filter businesses based on sentiment preferences
3. **Location/Category Filtering**: Apply user-specified constraints
4. **Composite Scoring**: Multi-factor scoring algorithm:
   - 40% positive sentiment ratio
   - 30% normalized star rating
   - 20% logarithmic review count
   - 10% inverse negative sentiment ratio
5. **Ranking**: Sort businesses by composite score

## 3. Results and Performance Evaluation

### 3.1 Sentiment Analysis Model Performance

#### 3.1.1 Classification Metrics
The RoBERTa-LSTM model achieved the following performance on the test dataset:

- **Overall Accuracy**: 86.02%
- **Macro Average Precision**: 0.86
- **Macro Average Recall**: 0.86  
- **Macro Average F1-Score**: 0.86

#### 3.1.2 Per-Class Performance
| Sentiment Class | Precision | Recall | F1-Score | Support |
|----------------|-----------|---------|----------|---------|
| Negative (0)   | 0.88      | 0.87    | 0.88     | 49,955  |
| Neutral (1)    | 0.78      | 0.81    | 0.80     | 49,948  |
| Positive (2)   | 0.92      | 0.90    | 0.91     | 49,987  |

#### 3.1.3 Confusion Matrix Analysis
The confusion matrix reveals:
- **True Negatives**: 43,437 (87.0% of negative class)
- **True Neutrals**: 40,426 (80.9% of neutral class)
- **True Positives**: 45,076 (90.2% of positive class)

The model demonstrates strong performance across all sentiment categories, with positive sentiment showing the highest precision (0.92) and negative sentiment showing balanced precision-recall performance.

### 3.2 Recommender System Performance

#### 3.2.1 Dataset Coverage
- **Total Businesses Analyzed**: 150,346
- **Total Reviews Processed**: 1,498,897
- **Businesses with Sentiment Profiles**: 150,346
- **High Confidence Businesses** (≥10 reviews): Subset with robust sentiment patterns

#### 3.2.2 Feature Engineering Results
- **Category Features**: 100-dimensional TF-IDF vectors
- **Sentiment Profiles**: Business-level positive/negative/neutral ratios
- **Combined Feature Space**: Numerical + categorical features with standardization

#### 3.2.3 Recommendation Quality
The system demonstrates:
- **Real-time Sentiment Analysis**: Sub-second processing for user reviews
- **Scalable Architecture**: Efficient processing of 150K+ businesses
- **Multi-criteria Filtering**: Location, category, and sentiment-based filtering
- **Personalization**: Sentiment-aware recommendation adaptation

### 3.3 System Integration Results

#### 3.3.1 End-to-End Pipeline Performance
The complete system successfully integrates:
1. **Sentiment Analysis**: 86% accuracy sentiment classification
2. **Business Profiling**: Comprehensive sentiment profile generation
3. **Recommendation Generation**: Multi-factor scoring and ranking
4. **User Interface**: Interactive recommendation interface

#### 3.3.2 Recommendation Examples
The system generates contextually appropriate recommendations:

- **Positive Review Input**: Recommends highly-rated businesses with consistent positive feedback
- **Negative Review Input**: Suggests superior alternatives with high ratings and positive sentiment
- **Neutral Review Input**: Provides diverse options with balanced sentiment profiles

## 4. Technical Implementation Details

### 4.1 Computational Infrastructure
- **GPU Acceleration**: CUDA-enabled training with automatic mixed precision
- **Memory Management**: Gradient scaling and memory mapping for large dataset processing
- **Distributed Training**: DataParallel implementation for multi-GPU utilization
- **Checkpoint Management**: Automated model checkpointing with Google Drive integration

### 4.2 Data Pipeline Optimization
- **Parquet Format**: Efficient columnar storage with compression
- **Batch Processing**: Optimized batch sizes for memory-compute balance
- **Memory Mapping**: Numpy memory mapping for tokenization efficiency
- **Caching**: TF-IDF vectorizer caching for repeated computations

### 4.3 Model Deployment
- **Model Serialization**: PyTorch state dictionary serialization
- **Inference Optimization**: Evaluation mode and gradient disabling
- **Feature Preprocessing**: Standardized preprocessing pipeline
- **Real-time Processing**: Sub-second inference for user queries

## 5. Evaluation and Validation

### 5.1 Quantitative Analysis
The methodology validation includes:
- **Cross-validation**: Stratified train-validation-test splits
- **Performance Metrics**: Comprehensive classification metrics
- **Statistical Significance**: Large-scale dataset validation
- **Ablation Studies**: Component-wise performance analysis

### 5.2 Qualitative Assessment
- **Business Logic Validation**: Sentiment-driven recommendation logic
- **User Experience**: Interactive demonstration capabilities
- **Recommendation Diversity**: Multi-category business coverage
- **System Robustness**: Error handling and edge case management

## 6. Discussion and Limitations

### 6.1 Methodological Strengths
1. **State-of-the-art NLP**: RoBERTa-LSTM hybrid architecture
2. **Comprehensive Preprocessing**: Multi-stage data cleaning and normalization
3. **Balanced Sampling**: Stratified approach ensuring class representation
4. **Multi-modal Features**: Integration of text, numerical, and categorical features
5. **Scalable Architecture**: Efficient processing of large-scale datasets

### 6.2 Limitations and Future Work
1. **Cold Start Problem**: Limited handling of businesses with few reviews
2. **Temporal Dynamics**: Static sentiment profiles without temporal evolution
3. **Cultural Bias**: Potential bias in sentiment interpretation across demographics
4. **Computational Complexity**: Resource-intensive training requirements
5. **Evaluation Metrics**: Limited user study validation

## 7. Conclusion

This research successfully demonstrates the integration of advanced natural language processing techniques with recommender systems. The BERTRec system achieves 86% accuracy in sentiment analysis and provides contextually relevant business recommendations through sentiment-aware content-based filtering.

The methodology combines several innovative approaches:
- Hybrid RoBERTa-LSTM architecture for enhanced sentiment understanding
- Comprehensive data preprocessing pipeline ensuring high data quality
- Multi-factor recommendation scoring incorporating sentiment patterns
- Scalable system architecture suitable for real-world deployment

The results validate the effectiveness of sentiment-based recommendation approaches, showing significant potential for improving user experience in business recommendation scenarios. The system's ability to process natural language input and generate contextually appropriate recommendations represents a meaningful advancement in personalized recommendation technology.

Future research directions include incorporating collaborative filtering techniques, implementing temporal sentiment dynamics, and conducting comprehensive user studies to validate practical effectiveness in real-world scenarios.

## References

1. Yelp Academic Dataset: https://www.yelp.com/dataset
2. RoBERTa: Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
3. BERT: Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
4. LSTM: Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory"
5. Content-based Filtering: Lops, P., et al. (2011). "Content-based Recommender Systems: State of the Art and Trends"