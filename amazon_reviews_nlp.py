"""
NLP with spaCy: Amazon Product Reviews Analysis
Goal:
- Perform Named Entity Recognition (NER) to extract product names and brands
- Analyze sentiment using a rule-based approach
"""

import spacy
from spacy import displacy
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
    }
]


class RuleBasedSentimentAnalyzer:
    """
    Rule-based sentiment analyzer using lexicon approach
    """
    
    def __init__(self):
        # Positive sentiment words
        self.positive_words = {
            'love', 'amazing', 'best', 'great', 'excellent', 'fantastic', 'phenomenal',
            'outstanding', 'perfect', 'wonderful', 'incredible', 'awesome', 'brilliant',
            'superb', 'beautiful', 'comfortable', 'highly recommend', 'worth', 'good',
            'nice', 'happy', 'satisfied', 'pleased', 'impressed', 'quality'
        }
        
        # Negative sentiment words
        self.negative_words = {
            'terrible', 'poor', 'bad', 'awful', 'horrible', 'worst', 'disappointed',
            'disappointing', 'useless', 'waste', 'broken', 'issues', 'problem', 'problems',
            'not recommend', 'would not', 'never', 'hate', 'regret', 'overpriced',
            'underwhelming', 'lags', 'crashes', 'drains'
        }
        
        # Intensifiers
        self.intensifiers = {
            'very', 'extremely', 'absolutely', 'really', 'incredibly', 'highly',
            'super', 'totally', 'completely'
        }
        
        # Negations
        self.negations = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere',
            'hardly', 'barely', 'scarcely', "n't", 'cannot', "can't", "won't"
        }
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using rule-based approach
        Returns: (sentiment_label, confidence_score, details)
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
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
        
        # Determine sentiment
        total_score = positive_score + negative_score
        
        if total_score == 0:
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
        
        details = {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'total_score': total_score
        }
        
        return sentiment, confidence, details


def load_spacy_model():
    """
    Load spaCy model for NER
    """
    try:
        print("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
        print("✓ Model loaded successfully\n")
        return nlp
    except OSError:
        print("✗ spaCy model not found. Installing...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
        print("✓ Model loaded successfully\n")
        return nlp


def extract_entities(nlp, text):
    """
    Extract named entities from text
    Focus on products (ORG) and brands
    """
    doc = nlp(text)
    
    entities = {
        'products': [],
        'brands': [],
        'organizations': [],
        'all_entities': []
    }
    
    for ent in doc.ents:
        entity_info = {
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        }
        
        entities['all_entities'].append(entity_info)
        
        # Classify entities
        if ent.label_ == 'ORG':
            entities['organizations'].append(ent.text)
            # Common tech brands
            brands = ['Apple', 'Samsung', 'Sony', 'Nike', 'Amazon', 'Dell', 
                     'Google', 'Bose', 'Canon', 'Microsoft']
            if any(brand in ent.text for brand in brands):
                entities['brands'].append(ent.text)
        
        elif ent.label_ == 'PRODUCT':
            entities['products'].append(ent.text)
    
    # Additional pattern matching for product names
    product_patterns = [
        r'iPhone \d+\s?\w*',
        r'Galaxy S\d+',
        r'WH-\d+\w+',
        r'Air Max \d+',
        r'Echo Dot \d+\w+ Gen',
        r'XPS \d+',
        r'Pixel \d+',
        r'QuietComfort \d+',
        r'EOS R\d+',
        r'Surface Pro \d+'
    ]
    
    for pattern in product_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            product = match.group()
            if product not in entities['products']:
                entities['products'].append(product)
    
    return entities


def analyze_reviews(reviews):
    """
    Analyze Amazon product reviews for NER and sentiment
    """
    print("="*80)
    print("NLP ANALYSIS: AMAZON PRODUCT REVIEWS")
    print("="*80)
    print()
    
    # Load spaCy model
    nlp = load_spacy_model()
    
    # Initialize sentiment analyzer
    sentiment_analyzer = RuleBasedSentimentAnalyzer()
    
    # Store results
    results = []
    all_brands = []
    all_products = []
    
    # Process each review
    for idx, review_data in enumerate(reviews, 1):
        review_text = review_data['review']
        rating = review_data['rating']
        
        print(f"{'='*80}")
        print(f"REVIEW #{idx} (Rating: {rating}/5)")
        print(f"{'='*80}")
        print(f"\n📝 Review Text:\n{review_text}\n")
        
        # Extract entities
        entities = extract_entities(nlp, review_text)
        
        print("🏷️  NAMED ENTITY RECOGNITION:")
        print("-" * 40)
        
        if entities['brands']:
            print(f"  Brands: {', '.join(set(entities['brands']))}")
            all_brands.extend(entities['brands'])
        else:
            print("  Brands: None detected")
        
        if entities['products']:
            print(f"  Products: {', '.join(set(entities['products']))}")
            all_products.extend(entities['products'])
        else:
            print("  Products: None detected")
        
        if entities['organizations']:
            print(f"  Organizations: {', '.join(set(entities['organizations']))}")
        
        if entities['all_entities']:
            print(f"\n  All Entities Detected:")
            for ent in entities['all_entities']:
                print(f"    - {ent['text']} ({ent['label']})")
        
        # Analyze sentiment
        sentiment, confidence, details = sentiment_analyzer.analyze_sentiment(review_text)
        
        print(f"\n💭 SENTIMENT ANALYSIS:")
        print("-" * 40)
        print(f"  Sentiment: {sentiment}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Positive Score: {details['positive_score']:.1f}")
        print(f"  Negative Score: {details['negative_score']:.1f}")
        
        # Sentiment indicator
        if sentiment == "Positive":
            indicator = "😊 POSITIVE"
        elif sentiment == "Negative":
            indicator = "😞 NEGATIVE"
        else:
            indicator = "😐 NEUTRAL"
        
        print(f"  Overall: {indicator}")
        
        # Store results
        results.append({
            'review_id': idx,
            'review': review_text,
            'rating': rating,
            'brands': list(set(entities['brands'])),
            'products': list(set(entities['products'])),
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_score': details['positive_score'],
            'negative_score': details['negative_score']
        })
        
        print()
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Brand frequency
    brand_counter = Counter(all_brands)
    print("\n📊 Most Mentioned Brands:")
    print("-" * 40)
    for brand, count in brand_counter.most_common(5):
        print(f"  {brand}: {count} mention(s)")
    
    # Product frequency
    product_counter = Counter(all_products)
    print("\n📊 Most Mentioned Products:")
    print("-" * 40)
    if product_counter:
        for product, count in product_counter.most_common(5):
            print(f"  {product}: {count} mention(s)")
    else:
        print("  No specific products detected")
    
    # Sentiment distribution
    sentiment_counts = Counter([r['sentiment'] for r in results])
    print("\n📊 Sentiment Distribution:")
    print("-" * 40)
    total = len(results)
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total) * 100
        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    
    # Average confidence
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    print(f"\n📊 Average Sentiment Confidence: {avg_confidence:.2%}")
    
    # Correlation between rating and sentiment
    print("\n📊 Rating vs Sentiment Analysis:")
    print("-" * 40)
    correct_predictions = 0
    for r in results:
        predicted_positive = r['sentiment'] == 'Positive'
        actual_positive = r['rating'] >= 4
        
        if predicted_positive == actual_positive:
            correct_predictions += 1
            match = "✓"
        else:
            match = "✗"
        
        print(f"  Review #{r['review_id']}: Rating {r['rating']}/5 → {r['sentiment']} {match}")
    
    accuracy = (correct_predictions / total) * 100
    print(f"\n  Sentiment-Rating Accuracy: {accuracy:.1f}%")
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("RESULTS DATAFRAME")
    print("="*80)
    print(df_results[['review_id', 'rating', 'sentiment', 'confidence', 'brands', 'products']].to_string())
    
    # Save results
    df_results.to_csv('amazon_reviews_analysis.csv', index=False)
    print("\n✓ Results saved to 'amazon_reviews_analysis.csv'")
    
    return results, df_results


def demonstrate_entity_visualization(nlp, text):
    """
    Demonstrate entity visualization using displaCy
    """
    print("\n" + "="*80)
    print("ENTITY VISUALIZATION EXAMPLE")
    print("="*80)
    print(f"\nSample Text:\n{text}\n")
    
    doc = nlp(text)
    
    print("Entities detected:")
    for ent in doc.ents:
        print(f"  - {ent.text:20} → {ent.label_:10} ({spacy.explain(ent.label_)})")
    
    # Generate HTML visualization
    html = displacy.render(doc, style="ent", page=True)
    
    with open("entity_visualization.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    print("\n✓ Entity visualization saved to 'entity_visualization.html'")


def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print("🔍 NLP WITH SPACY: AMAZON PRODUCT REVIEWS ANALYSIS")
    print("="*80)
    print("\nObjectives:")
    print("  1. Perform Named Entity Recognition (NER) to extract products and brands")
    print("  2. Analyze sentiment using rule-based approach")
    print("="*80)
    print()
    
    # Analyze reviews
    results, df_results = analyze_reviews(SAMPLE_REVIEWS)
    
    # Demonstrate entity visualization
    nlp = spacy.load("en_core_web_sm")
    sample_text = "The Apple iPhone 14 Pro and Samsung Galaxy S23 are competing flagship smartphones from major tech brands."
    demonstrate_entity_visualization(nlp, sample_text)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - amazon_reviews_analysis.csv (detailed results)")
    print("  - entity_visualization.html (entity visualization example)")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
