"""
NLP Analysis: Amazon Product Reviews
Goal:
- Perform Named Entity Recognition (NER) to extract product names and brands
- Analyze sentiment using a rule-based approach
- Works without spaCy (uses pattern matching and TextBlob)
"""

from textblob import TextBlob
import pandas as pd
from collections import Counter
import re

# Sample Amazon product reviews dataset
SAMPLE_REVIEWS = [
    {
        "review": "I absolutely love my new Apple iPhone 14 Pro! The camera quality is amazing and the battery lasts all day. Best purchase ever!",
        "rating": 5
    },
    {
        "review": "The Samsung Galaxy S23 is decent but overpriced. The screen is beautiful but battery life could be better.",
        "rating": 3
    },
    {
        "review": "Terrible experience with the Sony WH-1000XM5 headphones. Sound quality is poor and they broke after a week. Would not recommend.",
        "rating": 1
    },
    {
        "review": "The Nike Air Max 270 sneakers are incredibly comfortable! Perfect for running and everyday wear. Highly recommend!",
        "rating": 5
    },
    {
        "review": "Amazon Echo Dot 5th Gen is a fantastic smart speaker. Alexa responds quickly and the sound quality is great for the price.",
        "rating": 5
    },
    {
        "review": "The Dell XPS 13 laptop is okay. Good build quality but the keyboard feels cramped. Not worth the premium price.",
        "rating": 3
    },
    {
        "review": "Absolutely disappointed with the Google Pixel 7. Camera crashes frequently and battery drains too fast. Poor quality control.",
        "rating": 2
    },
    {
        "review": "The Bose QuietComfort 45 headphones are phenomenal! Noise cancellation is the best I've experienced. Worth every penny!",
        "rating": 5
    },
    {
        "review": "Canon EOS R6 is an excellent camera for professionals. Image quality is outstanding and autofocus is lightning fast.",
        "rating": 5
    },
    {
        "review": "The Microsoft Surface Pro 9 is underwhelming. Screen is nice but performance lags and it gets hot quickly. Expected better.",
        "rating": 2
    },
    {
        "review": "Love the Adidas Ultraboost 22 running shoes! Super comfortable and stylish. Great for marathon training.",
        "rating": 5
    },
    {
        "review": "The LG OLED TV delivers stunning picture quality. Netflix and gaming look incredible. A bit expensive but worth it.",
        "rating": 5
    }
]


class EntityExtractor:
    """
    Named Entity Recognition for products and brands using pattern matching
    """
    
    def __init__(self):
        # Known tech and product brands
        self.brands = {
            'Apple', 'Samsung', 'Sony', 'Nike', 'Amazon', 'Dell', 'Google',
            'Bose', 'Canon', 'Microsoft', 'Adidas', 'LG', 'HP', 'Lenovo',
            'Asus', 'Acer', 'Intel', 'AMD', 'Nvidia', 'Logitech', 'Razer',
            'JBL', 'Sennheiser', 'Philips', 'Panasonic', 'Xiaomi', 'OnePlus',
            'Huawei', 'Motorola', 'Nokia', 'Tesla', 'Ford', 'Toyota', 'Honda'
        }
        
        # Product patterns
        self.product_patterns = [
            # Smartphones
            r'iPhone \d+\s?(?:Pro|Plus|Max|Mini)?',
            r'Galaxy S\d+(?:\s?(?:Plus|\+|Ultra))?',
            r'Pixel \d+(?:\s?(?:Pro|XL|a))?',
            
            # Headphones/Audio
            r'WH-\d+\w+',
            r'QuietComfort \d+',
            r'AirPods(?:\s?(?:Pro|Max))?',
            
            # Shoes
            r'Air Max \d+',
            r'Ultraboost \d+',
            
            # Smart devices
            r'Echo Dot \d+(?:th|st|nd|rd)?\s?Gen',
            r'Echo \d+(?:th|st|nd|rd)?\s?Gen',
            r'Alexa',
            
            # Computers
            r'XPS \d+',
            r'MacBook(?:\s?(?:Pro|Air))?',
            r'Surface Pro \d+',
            r'ThinkPad \w+\d+',
            
            # Cameras
            r'EOS R\d+',
            r'Alpha \d+',
            r'Z\d+',
            
            # TVs
            r'OLED TV',
            r'QLED TV',
            r'\d+"?\s?(?:OLED|QLED|LED|LCD)',
        ]
        
        # Product type keywords
        self.product_types = {
            'laptop', 'computer', 'phone', 'smartphone', 'tablet', 'headphones',
            'earbuds', 'speaker', 'camera', 'tv', 'television', 'monitor',
            'keyboard', 'mouse', 'shoes', 'sneakers', 'watch', 'smartwatch'
        }
    
    def extract_entities(self, text):
        """
        Extract brands and products from text
        """
        entities = {
            'brands': [],
            'products': [],
            'product_types': []
        }
        
        # Extract brands (case-insensitive search)
        text_words = text.split()
        for brand in self.brands:
            # Use word boundaries for more accurate matching
            pattern = r'\b' + re.escape(brand) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                entities['brands'].append(brand)
        
        # Extract products using patterns
        for pattern in self.product_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                product = match.group().strip()
                if product and product not in entities['products']:
                    entities['products'].append(product)
        
        # Extract product types
        text_lower = text.lower()
        for product_type in self.product_types:
            if product_type in text_lower:
                entities['product_types'].append(product_type)
        
        # Remove duplicates while preserving order
        entities['brands'] = list(dict.fromkeys(entities['brands']))
        entities['products'] = list(dict.fromkeys(entities['products']))
        entities['product_types'] = list(dict.fromkeys(entities['product_types']))
        
        return entities


class RuleBasedSentimentAnalyzer:
    """
    Enhanced rule-based sentiment analyzer
    """
    
    def __init__(self):
        # Positive sentiment words
        self.positive_words = {
            'love', 'amazing', 'best', 'great', 'excellent', 'fantastic', 'phenomenal',
            'outstanding', 'perfect', 'wonderful', 'incredible', 'awesome', 'brilliant',
            'superb', 'beautiful', 'comfortable', 'highly recommend', 'worth', 'good',
            'nice', 'happy', 'satisfied', 'pleased', 'impressed', 'quality', 'stunning',
            'fast', 'quick', 'stylish', 'powerful'
        }
        
        # Negative sentiment words
        self.negative_words = {
            'terrible', 'poor', 'bad', 'awful', 'horrible', 'worst', 'disappointed',
            'disappointing', 'useless', 'waste', 'broken', 'issues', 'problem', 'problems',
            'not recommend', 'would not', 'never', 'hate', 'regret', 'overpriced',
            'underwhelming', 'lags', 'crashes', 'drains', 'cramped', 'expensive'
        }
        
        # Intensifiers
        self.intensifiers = {
            'very', 'extremely', 'absolutely', 'really', 'incredibly', 'highly',
            'super', 'totally', 'completely', 'so', 'too'
        }
        
        # Negations
        self.negations = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
            'hardly', 'barely', 'scarcely', "n't", 'cannot', "can't", "won't"
        }
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using both rule-based and TextBlob approaches
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Rule-based scoring
        positive_score = 0
        negative_score = 0
        
        for i, word in enumerate(words):
            # Check for negation in previous 3 words
            negation_context = False
            if i > 0:
                for j in range(max(0, i-3), i):
                    if words[j] in self.negations:
                        negation_context = True
                        break
            
            # Check for intensifier in previous 2 words
            intensifier_context = False
            if i > 0:
                for j in range(max(0, i-2), i):
                    if words[j] in self.intensifiers:
                        intensifier_context = True
                        break
            
            # Calculate score
            multiplier = 1.5 if intensifier_context else 1.0
            
            if word in self.positive_words:
                if negation_context:
                    negative_score += 1 * multiplier
                else:
                    positive_score += 1 * multiplier
            
            if word in self.negative_words:
                if negation_context:
                    positive_score += 1 * multiplier
                else:
                    negative_score += 1 * multiplier
        
        # Check for positive/negative phrases
        if 'not recommend' in text_lower or 'would not' in text_lower:
            negative_score += 2
        if 'highly recommend' in text_lower:
            positive_score += 2
        
        # TextBlob sentiment for additional insight
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity  # -1 to 1
        textblob_subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Combine rule-based and TextBlob
        total_score = positive_score + negative_score
        
        if total_score == 0:
            # Use TextBlob as fallback
            if textblob_polarity > 0.1:
                sentiment = "Positive"
                confidence = min(abs(textblob_polarity), 0.7)
            elif textblob_polarity < -0.1:
                sentiment = "Negative"
                confidence = min(abs(textblob_polarity), 0.7)
            else:
                sentiment = "Neutral"
                confidence = 0.5
        elif positive_score > negative_score:
            sentiment = "Positive"
            confidence = min(positive_score / max(total_score, 1), 1.0)
        elif negative_score > positive_score:
            sentiment = "Negative"
            confidence = min(negative_score / max(total_score, 1), 1.0)
        else:
            sentiment = "Neutral"
            confidence = 0.5
        
        # Boost confidence if both methods agree
        if (sentiment == "Positive" and textblob_polarity > 0.1) or \
           (sentiment == "Negative" and textblob_polarity < -0.1):
            confidence = min(confidence * 1.2, 1.0)
        
        details = {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'total_score': total_score,
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity
        }
        
        return sentiment, confidence, details


def analyze_reviews(reviews):
    """
    Analyze Amazon product reviews for NER and sentiment
    """
    print("="*80)
    print("NLP ANALYSIS: AMAZON PRODUCT REVIEWS")
    print("="*80)
    print()
    
    # Initialize extractors
    entity_extractor = EntityExtractor()
    sentiment_analyzer = RuleBasedSentimentAnalyzer()
    
    # Store results
    results = []
    all_brands = []
    all_products = []
    all_product_types = []
    
    # Process each review
    for idx, review_data in enumerate(reviews, 1):
        review_text = review_data['review']
        rating = review_data['rating']
        
        print(f"{'='*80}")
        print(f"REVIEW #{idx} (Rating: {'⭐' * rating})")
        print(f"{'='*80}")
        print(f"\n📝 Review Text:")
        print(f"   {review_text}\n")
        
        # Extract entities
        entities = entity_extractor.extract_entities(review_text)
        
        print("🏷️  NAMED ENTITY RECOGNITION (NER):")
        print("-" * 40)
        
        if entities['brands']:
            print(f"  📱 Brands Detected: {', '.join(entities['brands'])}")
            all_brands.extend(entities['brands'])
        else:
            print("  📱 Brands: None detected")
        
        if entities['products']:
            print(f"  🎯 Products Detected: {', '.join(entities['products'])}")
            all_products.extend(entities['products'])
        else:
            print("  🎯 Products: None detected")
        
        if entities['product_types']:
            print(f"  📦 Product Types: {', '.join(entities['product_types'])}")
            all_product_types.extend(entities['product_types'])
        
        # Analyze sentiment
        sentiment, confidence, details = sentiment_analyzer.analyze_sentiment(review_text)
        
        print(f"\n💭 SENTIMENT ANALYSIS:")
        print("-" * 40)
        print(f"  Sentiment: {sentiment}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Rule-based Scores:")
        print(f"    • Positive: {details['positive_score']:.1f}")
        print(f"    • Negative: {details['negative_score']:.1f}")
        print(f"  TextBlob Scores:")
        print(f"    • Polarity: {details['textblob_polarity']:.3f} (range: -1 to 1)")
        print(f"    • Subjectivity: {details['textblob_subjectivity']:.3f} (range: 0 to 1)")
        
        # Sentiment indicator
        if sentiment == "Positive":
            indicator = "😊 POSITIVE"
            color = "✓"
        elif sentiment == "Negative":
            indicator = "😞 NEGATIVE"
            color = "✗"
        else:
            indicator = "😐 NEUTRAL"
            color = "○"
        
        print(f"  {color} Overall: {indicator}")
        
        # Store results
        results.append({
            'review_id': idx,
            'review': review_text[:50] + '...' if len(review_text) > 50 else review_text,
            'full_review': review_text,
            'rating': rating,
            'brands': entities['brands'],
            'products': entities['products'],
            'product_types': entities['product_types'],
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_score': details['positive_score'],
            'negative_score': details['negative_score'],
            'textblob_polarity': details['textblob_polarity']
        })
        
        print()
    
    # Summary statistics
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    
    # Brand frequency
    brand_counter = Counter(all_brands)
    print("\n🏆 Most Mentioned Brands:")
    print("-" * 40)
    if brand_counter:
        for brand, count in brand_counter.most_common(10):
            bar = "█" * count
            print(f"  {brand:15} {bar} ({count})")
    else:
        print("  No brands detected")
    
    # Product frequency
    product_counter = Counter(all_products)
    print("\n🎯 Most Mentioned Products:")
    print("-" * 40)
    if product_counter:
        for product, count in product_counter.most_common(10):
            print(f"  • {product} ({count})")
    else:
        print("  No specific products detected")
    
    # Product types
    type_counter = Counter(all_product_types)
    print("\n📦 Product Categories:")
    print("-" * 40)
    if type_counter:
        for ptype, count in type_counter.most_common(10):
            print(f"  • {ptype}: {count}")
    
    # Sentiment distribution
    sentiment_counts = Counter([r['sentiment'] for r in results])
    print("\n💭 Sentiment Distribution:")
    print("-" * 40)
    total = len(results)
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 5)
        print(f"  {sentiment:10} {bar} {count} ({percentage:.1f}%)")
    
    # Average confidence
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    print(f"\n📈 Average Sentiment Confidence: {avg_confidence:.2%}")
    
    # Correlation between rating and sentiment
    print("\n🎯 Rating vs Sentiment Analysis:")
    print("-" * 40)
    correct_predictions = 0
    for r in results:
        predicted_positive = r['sentiment'] == 'Positive'
        actual_positive = r['rating'] >= 4
        
        if predicted_positive == actual_positive:
            correct_predictions += 1
            match = "✓"
        elif r['sentiment'] == 'Neutral':
            match = "○"
        else:
            match = "✗"
        
        stars = "⭐" * r['rating']
        print(f"  Review #{r['review_id']:2}: {stars:10} → {r['sentiment']:10} {match}")
    
    accuracy = (correct_predictions / total) * 100
    print(f"\n  ✓ Sentiment-Rating Accuracy: {accuracy:.1f}%")
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("📋 EXTRACTED ENTITIES SUMMARY")
    print("="*80)
    
    # Display sample of results
    display_df = df_results[['review_id', 'rating', 'brands', 'products', 'sentiment', 'confidence']].copy()
    display_df['brands'] = display_df['brands'].apply(lambda x: ', '.join(x) if x else 'None')
    display_df['products'] = display_df['products'].apply(lambda x: ', '.join(x) if x else 'None')
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
    
    print(display_df.to_string(index=False))
    
    # Save results
    df_results.to_csv('amazon_reviews_analysis.csv', index=False)
    print("\n✓ Results saved to 'amazon_reviews_analysis.csv'")
    
    return results, df_results


def display_sample_outputs(results):
    """
    Display sample outputs showing extracted entities and sentiment
    """
    print("\n" + "="*80)
    print("📤 SAMPLE OUTPUT: EXTRACTED ENTITIES AND SENTIMENT")
    print("="*80)
    
    # Select 3 diverse examples
    sample_indices = [0, 2, 7]  # Positive, Negative, Positive
    
    for idx in sample_indices:
        if idx < len(results):
            r = results[idx]
            print(f"\n{'─'*80}")
            print(f"Sample Review #{r['review_id']}:")
            print(f"{'─'*80}")
            print(f"Text: {r['full_review']}")
            print(f"\nExtracted Information:")
            print(f"  • Brands: {', '.join(r['brands']) if r['brands'] else 'None'}")
            print(f"  • Products: {', '.join(r['products']) if r['products'] else 'None'}")
            print(f"  • Product Types: {', '.join(r['product_types']) if r['product_types'] else 'None'}")
            print(f"  • Sentiment: {r['sentiment']} (Confidence: {r['confidence']:.2%})")
            print(f"  • Rating: {'⭐' * r['rating']}")


def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print("🔍 NLP WITH PATTERN MATCHING & TEXTBLOB")
    print("   Amazon Product Reviews Analysis")
    print("="*80)
    print("\nObjectives:")
    print("  ✓ Named Entity Recognition (NER) to extract products and brands")
    print("  ✓ Sentiment analysis using rule-based approach")
    print("="*80)
    print()
    
    # Analyze reviews
    results, df_results = analyze_reviews(SAMPLE_REVIEWS)
    
    # Display sample outputs
    display_sample_outputs(results)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nDeliverables:")
    print("  ✓ Code snippet with entity extraction and sentiment analysis")
    print("  ✓ Output showing extracted entities for each review")
    print("  ✓ Sentiment scores and classification")
    print("  ✓ CSV file with detailed results")
    print("\nGenerated files:")
    print("  📁 amazon_reviews_analysis.csv (detailed results)")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
