import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
import threading

# Thread-safe set for storing visited URLs
visited_urls = set()
lock = threading.Lock()  # Lock to synchronize access to the shared set

def fetch_and_extract(url, max_depth, current_depth):
    """
    Fetch the webpage content and extract URLs up to a specified depth.

    :param url: The URL to fetch and parse.
    :param max_depth: Maximum depth for recursion.
    :param current_depth: Current depth in the recursion.
    :return: A list of extracted URLs.
    """
    if current_depth > max_depth:
        return []

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return []

    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    urls = []

    for link in soup.find_all('a', href=True):
        full_url = urljoin(url, link['href'])  # Handle relative URLs
        # Ensure thread-safe access to the visited set
        with lock:
            if full_url not in visited_urls:
                visited_urls.add(full_url)
                urls.append(full_url)

    return urls

def process_url(url, max_depth, current_depth, executor):
    """
    Process a URL by fetching its content and recursively extracting links.

    :param url: The URL to process.
    :param max_depth: Maximum depth for recursion.
    :param current_depth: Current depth in the recursion.
    :param executor: ThreadPoolExecutor for managing threads.
    """
    i=1
    new_urls = fetch_and_extract(url, max_depth, current_depth)
    futures = []
    for new_url in new_urls:
        print(i)
        i+=1
        futures.append(executor.submit(process_url, new_url, max_depth, current_depth + 1, executor))
    # Wait for all futures to complete
    for future in futures:
        future.result()

def start_crawling(start_url, max_depth, num_threads):
    """
    Start the crawling process with multithreading.

    :param start_url: The initial URL to start crawling from.
    :param max_depth: Maximum depth for recursion.
    :param num_threads: Number of threads for the ThreadPoolExecutor.
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:

        process_url(start_url, max_depth, 0, executor)

# Initialize parameters
start_url = "https://mgit.ac.in"
max_depth = 3  # Limit the depth of recursion
num_threads = 10  # Number of threads to use

# Start the crawling process
start_crawling(start_url, max_depth, num_threads)

# Print unique URLs
print("\nUnique URLs found:")
for url in visited_urls:
    print(url)
