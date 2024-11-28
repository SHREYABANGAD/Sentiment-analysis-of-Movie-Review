import requests
from bs4 import BeautifulSoup
import re

# OMDb API Key
OMDB_API_KEY = '36650a58'


def fetch_similar_movies(letterboxd_url):
    """
    Fetch similar movies from Letterboxd and get their posters from OMDB
    """
    similar_movies = []
    
    try:
        similar_url = f"{letterboxd_url}/similar"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(similar_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        similar_grid = soup.find('ul', class_='poster-list')
        if similar_grid:
            movie_items = similar_grid.find_all('li', class_='poster-container')
            
            for item in movie_items[:6]:
                try:
                    film_poster = item.find('div', class_='film-poster')
                    if film_poster:
                        movie_title = film_poster.find('img')['alt']
                        movie_url = film_poster['data-target-link']
                        
                        # Get movie year from the URL if available
                        year = ''
                        if '-' in movie_url:
                            try:
                                year = movie_url.split('-')[-1]
                                if year.isdigit() and len(year) == 4:
                                    year = f"&y={year}"
                                else:
                                    year = ''
                            except:
                                year = ''
                        
                        # Fetch poster from OMDB API
                        omdb_url = f"http://www.omdbapi.com/?t={movie_title}{year}&apikey={OMDB_API_KEY}"
                        omdb_response = requests.get(omdb_url)
                        omdb_data = omdb_response.json()
                        
                        poster_url = omdb_data.get('Poster', 'N/A')
                        if poster_url == 'N/A':
                            # Use a default poster if OMDB doesn't have one
                            poster_url = "path/to/default/poster.jpg"
                        
                        similar_movies.append({
                            'title': movie_title,
                            'url': f"https://letterboxd.com{movie_url}",
                            'poster': poster_url,
                            'year': year.replace('&y=', '') if year else 'N/A'
                        })
                        
                except Exception as e:
                    print(f"Error processing similar movie: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error fetching similar movies: {e}")
    
    return similar_movies

def fetch_trending_movies():
    url = 'https://letterboxd.com/films/ajax/popular/size/large/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': 'https://letterboxd.com/films/popular/',
    }

    try:
        session = requests.Session()
        
        # Get the CSRF token first
        main_page = session.get('https://letterboxd.com/films/popular/', headers=headers)
        csrf_token = None
        if main_page.cookies.get('com.xk72.webparts.csrf'):
            csrf_token = main_page.cookies.get('com.xk72.webparts.csrf')
            headers['X-CSRF-Token'] = csrf_token

        # Make the AJAX request
        response = session.get(url, headers=headers)
        print("Response status:", response.status_code)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            trending_movies = []
            
            # Find all film posters
            posters = soup.find_all('div', class_='film-poster')
            
            for poster in posters[:12]:
                try:
                    # Get film data
                    title = poster.get('data-film-name')
                    if not title:
                        continue
                        
                    # Get image
                    img = poster.find('img')
                    if img:
                        image_url = img.get('src')
                        if image_url:
                            if image_url.startswith('//'):
                                image_url = 'https:' + image_url
                            # Convert to larger image size
                            image_url = image_url.replace('0-150-0-225', '0-230-0-345')
                    
                    # Get link
                    link = poster.find('a')
                    if link:
                        movie_slug = link.get('href', '').split('/film/')[-1].strip('/')
                    else:
                        movie_slug = title.lower().replace(' ', '-')
                    
                    movie_data = {
                        'title': title,
                        'link': movie_slug,
                        'image': image_url
                    }
                    
                    trending_movies.append(movie_data)
                    print(f"Added movie: {title}")
                    
                except Exception as e:
                    print(f"Error processing movie: {str(e)}")
                    continue
            
            return trending_movies
            
    except Exception as e:
        print(f"Error fetching trending movies: {str(e)}")
        return []


def get_omdb_data(movie_title):
    """Fetches data from OMDb API."""
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    return response.json()

def fetch_letterboxd_reviews(letterboxd_url):
    """Fetches reviews from Letterboxd."""
    response = requests.get(letterboxd_url)
    if response.status_code != 200:
        return []
    
    season_page = BeautifulSoup(response.content, 'html.parser')
    reviews = season_page.find_all('div', class_='body-text -prose collapsible-text')
    
    review_texts = []
    for review in reviews:
        review_text = review.get_text(strip=True)
        review_texts.append(review_text)
    
    return review_texts

def fetch_movie_reviews_and_details(movie_title, letterboxd_url):
    
    """Combines OMDb movie details with Letterboxd reviews."""
    # Fetch movie details from OMDb
    omdb_data = get_omdb_data(movie_title)
    
    # Fetch reviews from Letterboxd
    review_texts = fetch_letterboxd_reviews(letterboxd_url)

    similar_movies = fetch_similar_movies(letterboxd_url)

    # Organize movie details
    movie_details = {
        'title': omdb_data.get('Title', 'N/A'),
        'year': omdb_data.get('Year', 'N/A'),
        'rating': omdb_data.get('imdbRating', 'N/A'),
        'genres': omdb_data.get('Genre', 'N/A'),
        'box_office': omdb_data.get('BoxOffice', 'N/A'),
        'release_date': omdb_data.get('Released', 'N/A'),
        'plot': omdb_data.get('Plot', 'N/A'),
        'actor': omdb_data.get('Actors', 'N/A'),
        'director': omdb_data.get('Director', 'N/A'),
        'poster': omdb_data.get('Poster', 'N/A'),
        'language': omdb_data.get('Language', 'N/A'),
        'country': omdb_data.get('Country', 'N/A'),
        'awards': omdb_data.get('Awards', 'N/A'),
        'writer': omdb_data.get('Writer', 'N/A')
    }

    return review_texts, movie_details, similar_movies
