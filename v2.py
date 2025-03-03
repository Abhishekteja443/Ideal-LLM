import requests
from bs4 import BeautifulSoup
from collections import OrderedDict
import ollama
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime, timezone
import time
from xml.etree import ElementTree
from typing import List
import logging
import re
import gc
import psutil
import chromadb.config

settings = chromadb.config.Settings(
    allow_reset=True,
    persist_directory="./chromadb_store",
    is_persistent=True
)


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Change to INFO or ERROR in production
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="v2.log"
)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chromadb_store",settings=settings)
collection = client.get_or_create_collection(name="web_scraped_data")


def fetch_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """
    Recursively fetches all URLs from a sitemap or nested sitemaps.
    
    Args:
        sitemap_url (str): The root or nested sitemap URL.
    
    Returns:
        List[str]: List of all URLs to be crawled.
    """
    try:
        logging.info(f"Fetching URLs from sitemap: {sitemap_url}")
        response = requests.get(sitemap_url)
        response.raise_for_status()  # Raise exception for non-2xx responses
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Check for nested sitemaps
        sitemap_elements = root.findall('.//ns:sitemap/ns:loc', namespace)
        if sitemap_elements:
            nested_urls = [fetch_urls_from_sitemap(sitemap.text) for sitemap in sitemap_elements]
            return [url for sublist in nested_urls for url in sublist]
        
        # Otherwise, extract all URLs
        url_elements = root.findall('.//ns:url/ns:loc', namespace)
        return set(url.text for url in url_elements)
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching sitemap {sitemap_url}: {e}")
        return set()
    except ElementTree.ParseError as e:
        logging.error(f"Error parsing sitemap XML {sitemap_url}: {e}")
        return set()
    except Exception as e:
        logging.error(f"Unexpected error while fetching sitemap {sitemap_url}: {e}")
        return set()


def web_scrape_url(url):
    try:
        logging.info(f"Scraping URL: {url}")
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")

        scraped_data = []
        seen_texts = set()
        
        # Extract only relevant sections (modify the selector if needed)
        for tag in soup.find_all(["p", "span", "h1", "h2", "h3", "h4", "h5", "h6", "a","li","tr"]):
            text = tag.get_text(strip=True)
        
            # Handling anchor tags separately for links
            if tag.name == "a" and tag.get("href"):
                href = tag.get("href")
                icon = tag.find("i")
                if icon:
                    text=icon.get("class", [])
                    text= " ".join(text)
                    pattern = r"fa-([a-zA-Z0-9\-]+)"
                    text=re.findall(pattern, text)
                    text= " ".join(text)
                if text and (text, href) not in seen_texts:
                    seen_texts.add((text, href))
                    scraped_data.append(f"{text}: {href}")
            else:
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    scraped_data.append(text)

        return scraped_data

    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping URL {url}: {e}")
        return set()
    except Exception as e:
        logging.error(f"Unexpected error while scraping {url}: {e}")
        return set()


def chunk_text(text, chunk_size=800, chunk_overlap=400):
    try:
        logging.info("Chunking text into smaller parts")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        logging.info(f"Text split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logging.error(f"Error chunking text: {e}")
        return []


def check_and_delete_chunks(url):
    try:
        # Check if any chunks exist for the given source_url
        existing_chunks = collection.get(where={"source_url": url})
        x=existing_chunks["ids"]
        for i in x:
            print(i)
        if existing_chunks and "ids" in existing_chunks and existing_chunks["ids"]:
            logging.info(f"Found {len(existing_chunks['ids'])} existing chunks for {url}. Deleting...")
            print("deleting context")
            for i in x:
                time.sleep(10)
                collection.delete(ids=existing_chunks["ids"])
                
            logging.info(f"Deleted all chunks for {url}.")
            return True  # Indicate that chunks were found and deleted
        else:
            logging.info(f"No existing chunks found for {url}. Proceeding with insertion.")
            return False  # No chunks existed

    except Exception as e:
        # logging.error(f"Error while checking/deleting chunks for {url}: {e}")
        return False

def process_urls(urls):
    try:
        logging.info(f"Processing {len(urls)} URLs")
        for url in urls:
            logging.info(f"Processing URL: {url}")

            # check_and_delete_chunks(url)
            
            # Step 1: Scrape the URL
            data = web_scrape_url(url)

            if not data:
                logging.warning(f"No data found for {url}. Skipping...")
                continue  # Skip if no data is returned
            
            # Step 2: Process chunks
            scraped_data = {
                "source_url": url,
                "text": f"""{data}""",
            }

            chunks = chunk_text(scraped_data["text"])

            if not chunks:
                logging.warning(f"No chunks created for {url}. Skipping...")
                continue  # Skip if no chunks are created

            # Step 3: Embed and store each chunk in ChromaDB
            for i, chunk in enumerate(chunks):
                # print(i)
                chunk_id = f"{scraped_data['source_url']}_chunk_{i+1}"
                
                # Generate embedding using Ollama's Nomic Embeddings
                embedding = ollama.embeddings(model="nomic-embed-text:latest", prompt=chunk)["embedding"]
                # print(embedding)
                if not embedding:
                    logging.error(f"Failed to generate embedding for chunk {chunk_id}")
                    continue  # Skip if embedding generation fails
            
                # Store in ChromaDB
                collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "source_url": scraped_data["source_url"],
                        "chunk_size": len(chunk),
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                    }]
                )
                gc.collect()

                logging.info(f"Chunk {chunk_id} stored successfully in ChromaDB.")
                
        logging.info("Processing completed.")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")


def main():
    main_url = "https://mgit.ac.in/library_post-sitemap.xml"  # Replace with your main URL or sitemap URL
    if 'sitemap.xml' in main_url:
        urls = fetch_urls_from_sitemap(main_url)
    elif main_url == "":
        logging.error("Invalid URL entered.")
        return
    else:
        urls = [main_url]  # If it's a direct URL instead of a sitemap
    logging.info(f"Memory Usage Before Processing: {psutil.virtual_memory().percent}%")
    try:
        process_urls(urls)
    except Exception as e:
        logging.error(f"Kernel Crash Debug: {e}")

    logging.info(f"Memory Usage After Processing: {psutil.virtual_memory().percent}%")
    


if __name__ == "__main__":
    main()
