from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib
import plotly.graph_objects as go
matplotlib.use('Agg')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def analyze_sentiment(reviews):
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    polarity_scores = []
    positive_reviews = []
    neutral_reviews = []
    negative_reviews = []
    
    for review in reviews:
        analysis = TextBlob(review)
        polarity_scores.append(analysis.sentiment.polarity)
        if analysis.sentiment.polarity > 0:
            sentiments['positive'] += 1
            positive_reviews.append(review)
        elif analysis.sentiment.polarity == 0:
            sentiments['neutral'] += 1
            neutral_reviews.append(review)
        else:
            sentiments['negative'] += 1
            negative_reviews.append(review)
    
    positive_keywords = extract_keywords(positive_reviews, 'positive')
    negative_keywords = extract_keywords(negative_reviews, 'negative')
    
    return sentiments, positive_reviews, neutral_reviews, negative_reviews, polarity_scores, positive_keywords, negative_keywords

def extract_keywords(reviews, sentiment, sentiment_labels=['positive', 'neutral', 'negative']):
    stop_words = set(stopwords.words('english'))
    
    # Create a dictionary to store keywords for each sentiment label
    sentiment_keywords = {label: [] for label in sentiment_labels}
    
    # Loop through the reviews and categorize them based on sentiment
    for review, sentiment_label in zip(reviews, sentiment):
        # Ensure the sentiment is valid (e.g., positive, neutral, negative)
        if sentiment_label not in sentiment_labels:
            continue
        
        # Tokenize the review and remove stop words and non-alphanumeric tokens
        tokens = word_tokenize(review.lower())
        filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
        
        # Add filtered words to the appropriate sentiment category
        sentiment_keywords[sentiment_label].extend(filtered_words)
    
    # Extract top 10 keywords for each sentiment
    top_keywords = {}
    for label, words in sentiment_keywords.items():
        top_keywords[label] = Counter(words).most_common(10)  # Get the 10 most common words
    
    return top_keywords

# Generate a word cloud from reviews
def generate_word_cloud(reviews):
    all_reviews = ' '.join(reviews)
    
    if all_reviews.strip():
        wordcloud = WordCloud(width=800, height=400).generate(all_reviews)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('static/wordcloud.png')

# Create sentiment visualizations
def create_visualizations(sentiments, polarity_scores, movie_details, positive_reviews, neutral_reviews, negative_reviews):
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [sentiments['positive'], sentiments['neutral'], sentiments['negative']]
    colors = ['#4CAF50', '#FFC107', '#F44336']

    sample_data = {
    'Positive': positive_reviews,
    'Neutral': neutral_reviews,
    'Negative': negative_reviews}
    hover_text = [
    f"{label}<br>" + "<br>".join(sample_data[label]) for label in labels]

    # Pie chart for sentiment distribution
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,colors=colors)
    plt.title(f"Sentiment Analysis for {movie_details['title']}")
    plt.savefig('static/sentiment_pie.png')
    fig = go.Figure(data=[go.Pie(
    labels=labels,
    values=sizes,
    text=hover_text,
    hoverinfo="text+percent",
    textinfo="label+percent",
    marker=dict(colors=colors),
    hoverlabel=dict(
        font=dict(size=8),  # Change hover text size
        align="auto"  # Adjust hover text alignment
    ))])
    # Update layout
    fig.update_layout(
    title=f"Sentiment Analysis for {movie_details['title']}",
    height=500, width=500,  font=dict(size=8)
    )
    #Save the figure
    fig.write_html("templates/sentiment_pie.html")

    # Bar chart for sentiment count
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=sizes, palette='coolwarm')
    plt.title(f"Sentiment Count for {movie_details['title']}")
    plt.ylabel("Count")
    plt.savefig('static/sentiment_bar.png')

    # Histogram for polarity distribution
    plt.figure(figsize=(8, 6))
    plt.hist(polarity_scores, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Sentiment Polarity Distribution for {movie_details['title']}")
    plt.xlabel("Polarity Score")
    plt.ylabel("Frequency")
    plt.savefig('static/sentiment_histogram.png')