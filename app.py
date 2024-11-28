from flask import Flask, render_template, request, redirect, url_for
from imdb_scraper import fetch_movie_reviews_and_details, fetch_trending_movies
from sentiment_analysis import analyze_sentiment, generate_word_cloud, create_visualizations
import json

app = Flask(__name__)

FALLBACK_TRENDING = [
    {
        'title': 'Oppenheimer',
        'link': 'oppenheimer',
        'image': 'https://a.ltrbxd.com/resized/film-poster/7/8/4/3/2/8/784328-oppenheimer-0-1000-0-1500-crop.jpg?v=e3c6e7a32c'
    },
    {
        'title': 'Barbie',
        'link': 'barbie',
        'image': 'https://a.ltrbxd.com/resized/film-poster/2/7/7/0/6/4/277064-barbie-0-230-0-345-crop.jpg?v=1b83dc7a71'
    },
    {
        'title': 'Fight Club',
        'link': 'fight-club',
        'image': 'https://a.ltrbxd.com/resized/film-poster/5/1/5/6/8/51568-fight-club-0-1000-0-1500-crop.jpg?v=768b32dfa4'
    },
    {
        'title': 'Parasite',
        'link': 'parasite',
        'image': 'https://a.ltrbxd.com/resized/film-poster/4/2/6/4/0/6/426406-parasite-0-1000-0-1500-crop.jpg?v=8f5653f710'
    },
    {
        'title': 'Everything Everywhere All at Once',
        'link': 'everything-everywhere-all-at-once',
        'image': 'https://a.ltrbxd.com/resized/film-poster/4/7/4/4/7/4/474474-everything-everywhere-all-at-once-0-1000-0-1500-crop.jpg?v=281f1a041e'
    },
    {
        'title': 'La La Land',
        'link': 'la-la-land',
        'image': 'https://a.ltrbxd.com/resized/film-poster/2/4/0/3/4/4/240344-la-la-land-0-1000-0-1500-crop.jpg?v=053670ff84'
    },
    {
        'title': 'Joker',
        'link': 'joker',
        'image': 'https://a.ltrbxd.com/resized/film-poster/4/0/6/7/7/5/406775-joker-0-1000-0-1500-crop.jpg?v=e4ea7f98cc '
    },
    {
        'title': 'Whiplash',
        'link': 'whiplash',
        'image': 'https://a.ltrbxd.com/resized/sm/upload/cl/dn/kr/f1/4C9LHDxMsoYI0S3iMPZdm3Oevwo-0-1000-0-1500-crop.jpg?v=d13ea36528'
    },
    {
        'title': 'Inception',
        'link': 'inception',
        'image': 'https://a.ltrbxd.com/resized/sm/upload/sv/95/s9/4j/inception-0-1000-0-1500-crop.jpg?v=30d7224316'
    },
    {
        'title': 'Spider-Man: Into the Spider-Verse',
        'link': 'spider-man-into-the-spider-verse',
        'image': 'https://a.ltrbxd.com/resized/film-poster/2/5/1/9/4/3/251943-spider-man-into-the-spider-verse-0-1000-0-1500-crop.jpg?v=538fe0ada6'
    },
    {
        'title': 'The Batman',
        'link': 'the-Batman',
        'image': 'https://a.ltrbxd.com/resized/film-poster/3/4/8/9/1/4/348914-the-batman-0-1000-0-1500-crop.jpg?v=ec12a8b7ce'
    },
    {
        'title': 'The Shawshank Redemption',
        'link': 'the-shawshank-redemption',
        'image': 'https://a.ltrbxd.com/resized/sm/upload/7l/hn/46/uz/zGINvGjdlO6TJRu9wESQvWlOKVT-0-1000-0-1500-crop.jpg?v=8736d1c395'
    },
    # Add more fallback movies as needed
]

@app.route('/', methods=['GET', 'POST'])
def index():
    trending_movies = fetch_trending_movies()
    print(trending_movies)

  
        
    if not trending_movies:
        print("Both methods failed, using fallback data")
        trending_movies = FALLBACK_TRENDING

    if request.method == 'POST':
        movie_title = request.form['letterboxd_url']
        letterboxd_url = request.form['letterboxd_url']
        letterboxd_url = letterboxd_url.replace(' ', '-').lower()

        letterboxd_urls = 'https://letterboxd.com/film/'+letterboxd_url
        
        reviews, movie_details,similar_movies = fetch_movie_reviews_and_details(movie_title, letterboxd_urls)
        
        print(similar_movies)
        if reviews:
            sentiments, positive_reviews, neutral_reviews, negative_reviews, polarity_scores, positive_keywords, negative_keywords = analyze_sentiment(reviews)
            generate_word_cloud(reviews)
            create_visualizations(sentiments, polarity_scores, movie_details, positive_reviews, neutral_reviews, negative_reviews)
            
            positive_keywords_json = json.dumps(dict(positive_keywords))
            negative_keywords_json = json.dumps(dict(negative_keywords))
            
            return redirect(url_for('results', 
                                    movie=movie_details['title'],
                                    year=movie_details['year'],
                                    rating=movie_details['rating'],
                                    genres=movie_details['genres'],
                                    plot=movie_details['plot'],
                                    actors=movie_details['actor'],
                                    poster=movie_details['poster'],
                                    language=movie_details['language'],
                                    country=movie_details['country'],
                                    writer=movie_details['writer'],
                                    awards=movie_details['awards'],
                                    director=movie_details['director'],
                                    box_office=movie_details['box_office'],
                                    release_date=movie_details['release_date'],
                                    positive=len(positive_reviews),
                                    neutral=len(neutral_reviews),
                                    negative=len(negative_reviews),
                                    total=len(reviews),
                                    positive_keywords=positive_keywords_json,
                                    negative_keywords=negative_keywords_json,
                                    similar_movies=json.dumps(similar_movies)

                                   ))
        else:
            return render_template('index.html', error="No reviews found for this movie.", trending_movies=trending_movies)
    
    return render_template('index.html', trending_movies=trending_movies)

@app.route('/interactive')
def interactive_chart():
    return render_template('sentiment_pie.html')


@app.route('/results')
def results():
    try:
        movie = request.args.get('movie')
        year = request.args.get('year')
        rating = request.args.get('rating')
        plot = request.args.get('plot')
        director = request.args.get('director')
        poster = request.args.get('poster')
        language = request.args.get('language')
        writer = request.args.get('writer')
        awards = request.args.get('awards')
        country = request.args.get('country')
        genres = request.args.get('genres')
        box_office = request.args.get('box_office')
        release_date = request.args.get('release_date')
        positive = int(request.args.get('positive'))
        neutral = int(request.args.get('neutral'))
        negative = int(request.args.get('negative'))
        total = int(request.args.get('total'))
        actors = request.args.get('actors')
        
        positive_keywords = json.loads(request.args.get('positive_keywords'))
        negative_keywords = json.loads(request.args.get('negative_keywords'))
        
        positive_keywords = [(word, count) for word, count in positive_keywords.items()]
        negative_keywords = [(word, count) for word, count in negative_keywords.items()]

        similar_movies = json.loads(request.args.get('similar_movies'))


        return render_template('results.html', 
                               movie=movie, 
                               plot=plot,
                               actors=actors,
                               director=director,
                               poster=poster,
                               language=language,
                               country=country,
                               writer=writer,
                               awards=awards,
                               year=year,
                               rating=rating,
                               genres=genres,
                               box_office=box_office,
                               release_date=release_date,
                               positive=positive, 
                               neutral=neutral, 
                               negative=negative, 
                               total=total,
                               positive_keywords=positive_keywords,
                               negative_keywords=negative_keywords,
                               similar_movies=similar_movies)
    except Exception as e:
        print(f"Error in results route: {str(e)}")
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)