# Sentiment-Based Business Recommender System

A comprehensive machine learning system that combines BERT-LSTM sentiment analysis with content-based recommendation algorithms to provide intelligent business recommendations based on user review sentiment patterns.

## 🎯 Project Overview

This project integrates a high-performance sentiment analysis model (86% F1-score) with a sophisticated recommender system to analyze user reviews and suggest similar businesses based on sentiment patterns, business features, and user preferences.

## 📚 Notebook Series

The project is organized into 7 numbered Jupyter notebooks following a clear progression:

### Core Data Processing
- **`01_sampling.ipynb`** - Data sampling and initial preprocessing from Yelp dataset
- **`02_data_preprocessing.ipynb`** - Text cleaning, feature engineering, and data preparation

### Sentiment Analysis Model
- **`03_bert_lstm_model_building.ipynb`** - BERT-LSTM model architecture and training
- **`04_bert_lstm_testing.ipynb`** - Model evaluation, testing, and performance analysis

### Recommender System
- **`05_sentiment_based_recommender_system.ipynb`** - Core recommender system implementation
- **`06_recommender_system_demo.ipynb`** - Interactive demonstrations and visualizations
- **`07_recommender_utils.ipynb`** - Utilities, testing, and deployment tools

## 🔧 System Architecture

### Sentiment Analysis Component
- **Model**: RoBERTa-LSTM hybrid architecture
- **Performance**: 86% F1-score on sentiment classification
- **Classes**: Negative (0), Neutral (1), Positive (2)
- **Input**: Raw text reviews
- **Output**: Sentiment label + confidence score

### Recommendation Engine
- **Type**: Content-based filtering with sentiment integration
- **Features**: Business categories, ratings, sentiment profiles, location
- **Scoring**: Multi-factor composite algorithm
- **Filters**: Location, category, sentiment preference
- **Output**: Ranked business recommendations

## 🚀 Key Features

- **Real-time Sentiment Analysis**: Instant sentiment prediction for user reviews
- **Intelligent Recommendations**: Business suggestions based on sentiment patterns
- **Multi-factor Scoring**: Combines sentiment, ratings, reviews, and similarity
- **Location & Category Filtering**: Targeted recommendations by geography and business type
- **Interactive Interface**: User-friendly demonstration system
- **Performance Monitoring**: Built-in benchmarking and health checks
- **Scalable Architecture**: Handles large datasets efficiently

## 📊 Dataset Information

- **Source**: Yelp Academic Dataset
- **Businesses**: 150,000+ analyzed
- **Reviews**: 1.5M processed (down from 6M original)
- **Coverage**: Multiple cities and business categories
- **Features**: Text reviews, ratings, business metadata, categories

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch, Transformers (RoBERTa)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Text Processing**: NLTK, regex, TF-IDF
- **Visualization**: matplotlib, seaborn
- **Data Storage**: Parquet, JSON
- **Development**: Jupyter Notebooks

## 📈 Performance Metrics

### Sentiment Model
- **Accuracy**: 86.0%
- **F1-Score**: 86.0%
- **Processing Speed**: ~50 reviews/second
- **Model Size**: Optimized for production

### Recommendation System
- **Response Time**: <1 second per request
- **Coverage**: 100% of businesses with reviews
- **Scalability**: Handles 150K+ businesses
- **Accuracy**: Content-based similarity matching

## 🎮 Usage Examples

### Basic Sentiment Analysis
```python
# Analyze sentiment of a review
sentiment, confidence = recommender.predict_user_sentiment(
    "Amazing food and excellent service! Highly recommend!"
)
# Output: ('positive', 0.95)
```

### Get Recommendations
```python
# Get business recommendations based on review
recommendations = recommend_businesses_from_review(
    user_review="Great coffee shop with friendly staff",
    user_location="Las Vegas",
    category_preference="Coffee & Tea",
    top_k=5
)
```

## 🔍 System Components

### Data Processing Pipeline
1. **Sampling**: Efficient data reduction from 6M to 1.5M reviews
2. **Preprocessing**: Text cleaning, tokenization, feature extraction
3. **Feature Engineering**: Business profiles, sentiment aggregation

### Model Training Pipeline
1. **Architecture**: RoBERTa encoder + BiLSTM + Classification head
2. **Training**: Supervised learning on labeled sentiment data
3. **Evaluation**: Comprehensive testing and validation

### Recommendation Pipeline
1. **Sentiment Analysis**: Real-time review sentiment prediction
2. **Feature Matching**: Business similarity calculation
3. **Filtering**: Location and category constraints
4. **Ranking**: Multi-factor scoring algorithm

## 🧪 Testing & Validation

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end system validation
- **Performance Benchmarks**: Speed and accuracy metrics
- **Health Checks**: System status monitoring

## 🚀 Deployment Ready

The system includes comprehensive deployment utilities:
- **Export Functions**: Save all components for production
- **Performance Monitoring**: Real-time system metrics
- **Health Checks**: Automated system validation
- **Batch Processing**: Efficient large-scale operations

## 📋 Requirements

```
torch>=1.9.0
transformers>=4.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🎯 Future Enhancements

- **Collaborative Filtering**: User-based recommendations
- **Deep Learning Embeddings**: Advanced business representations
- **Real-time Streaming**: Live review processing
- **Mobile App**: User-facing application
- **A/B Testing**: Recommendation algorithm optimization

## 📄 License

This project is developed for educational and research purposes.

## 🤝 Contributing

The system is designed with modularity and extensibility in mind. Each notebook can be run independently, and the architecture supports easy integration of new features.

---

**Built with ❤️ using state-of-the-art NLP and recommendation algorithms**
