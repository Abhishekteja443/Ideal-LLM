import requests
from bs4 import BeautifulSoup
from collections import OrderedDict
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime, timezone
import time
from xml.etree import ElementTree
from typing import List, Dict, Any
import logging
import re
import gc
import psutil
import os
import json
import numpy as np
from langchain_community.vectorstores import FAISS
import faiss

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="v3.log"
)

# Directory to store FAISS index and metadata
FAISS_INDEX_PATH = "./faiss_store"
METADATA_PATH = os.path.join(FAISS_INDEX_PATH, "metadata.json")

# Create directory if it doesn't exist
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

# Initialize or load metadata
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, 'r') as f:
        try:
            url_to_chunks = json.load(f)
        except json.JSONDecodeError:
            url_to_chunks = {}
else:
    url_to_chunks = {}

# Initialize FAISS index
index = None
all_documents = []
all_embeddings = []
all_metadatas = []
all_ids = []

def save_metadata():
    """Save metadata tracking URL to chunk mappings"""
    with open(METADATA_PATH, 'w') as f:
        json.dump(url_to_chunks, f)

def save_index():
    """Save the FAISS index and related data"""
    global index, all_documents, all_embeddings, all_metadatas, all_ids
    
    if len(all_documents) > 0:
        # Create a dictionary to store all necessary data
        index_data = {
            "documents": all_documents,
            "embeddings": all_embeddings,
            "metadatas": all_metadatas,
            "ids": all_ids
        }
        
        # Save to disk
        with open(os.path.join(FAISS_INDEX_PATH, "index_data.json"), 'w') as f:
            json.dump(index_data, f)
        
        # Create and save FAISS index
        if index is None:
            dimension = len(all_embeddings[0])
            index = faiss.IndexFlatL2(dimension)
        
        index.add(np.array(all_embeddings, dtype=np.float32))
        faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))
        
        logging.info(f"Saved FAISS index with {len(all_documents)} documents")
    else:
        logging.warning("No documents to save")

def load_index():
    """Load the FAISS index and related data"""
    global index, all_documents, all_embeddings, all_metadatas, all_ids
    
    if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index_data.json")) and \
       os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        
        # Load document data
        with open(os.path.join(FAISS_INDEX_PATH, "index_data.json"), 'r') as f:
            index_data = json.load(f)
            
        all_documents = index_data["documents"]
        all_embeddings = index_data["embeddings"]
        all_metadatas = index_data["metadatas"]
        all_ids = index_data["ids"]
        
        index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, "index.faiss"))
        
        logging.info(f"Loaded FAISS index with {len(all_documents)} documents")
        return True
    
    else:
        logging.info("No existing index found. Creating new...")
        all_documents = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        return False

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
        for tag in soup.find_all(["p", "span", "h1", "h2", "h3", "h4", "h5", "h6", "a", "li", "tr"]):
            text = tag.get_text(strip=True)
        
            # Handling anchor tags separately for links
            if tag.name == "a" and tag.get("href"):
                href = tag.get("href")
                icon = tag.find("i")
                if icon:
                    text = icon.get("class", [])
                    text = " ".join(text)
                    pattern = r"fa-([a-zA-Z0-9\-]+)"
                    text = re.findall(pattern, text)
                    text = " ".join(text)
                if text and (text, href) not in seen_texts:
                    seen_texts.add((text, href))
                    scraped_data.append(f"{text}: {href}")
            else:
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    scraped_data.append(text)
        scraped_data=" ".join(scraped_data)
        return scraped_data

    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping URL {url}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error while scraping {url}: {e}")
        return []

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
    """Delete all chunks associated with a specific URL"""
    global all_documents, all_embeddings, all_metadatas, all_ids
    
    try:
        if url in url_to_chunks and url_to_chunks[url]:
            chunk_ids = url_to_chunks[url]
            logging.info(f"Found {len(chunk_ids)} existing chunks for {url}. Deleting...")
            
            # Create new lists without the chunks to delete
            new_documents = []
            new_embeddings = []
            new_metadatas = []
            new_ids = []
            
            for i, doc_id in enumerate(all_ids):
                if doc_id not in chunk_ids:
                    new_documents.append(all_documents[i])
                    new_embeddings.append(all_embeddings[i])
                    new_metadatas.append(all_metadatas[i])
                    new_ids.append(doc_id)
            
            # Update the global lists
            all_documents = new_documents
            all_embeddings = new_embeddings
            all_metadatas = new_metadatas
            all_ids = new_ids
            
            # Remove the URL from metadata tracking
            del url_to_chunks[url]
            save_metadata()
            
            # Recreate the index from scratch (simpler than deleting from existing index)
            if len(all_embeddings) > 0:
                dimension = len(all_embeddings[0])
                global index
                index = faiss.IndexFlatL2(dimension)
                index.add(np.array(all_embeddings, dtype=np.float32))
            else:
                index = None
            
            logging.info(f"Deleted all chunks for {url}.")
            return True
        else:
            logging.info(f"No existing chunks found for {url}. Proceeding with insertion.")
            return False
    
    except Exception as e:
        logging.error(f"Error while checking/deleting chunks for {url}: {e}")
        return False

def process_urls(urls, batch_size=10):
    """Process URLs in batches to manage memory usage"""
    global all_documents, all_embeddings, all_metadatas, all_ids
    
    try:
        total_urls = len(urls)
        logging.info(f"Processing {total_urls} URLs in batches of {batch_size}")
        
        for i in range(0, total_urls, batch_size):
            batch_urls = list(urls)[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(total_urls+batch_size-1)//batch_size}")
            
            for url in batch_urls:
                process_single_url(url)
                
                # Force garbage collection after each URL
                gc.collect()
                
                # Log memory usage
                mem_usage = psutil.virtual_memory().percent
                logging.info(f"Memory usage after processing {url}: {mem_usage}%")
                
                # Add a small delay between URLs to be polite
                time.sleep(2)
            
            # Save progress after each batch
            save_index()
            save_metadata()
            
            # Add a longer delay between batches
            time.sleep(5)
        
        logging.info("All URLs processed successfully.")
    
    except Exception as e:
        logging.error(f"An error occurred during batch processing: {e}")
        # Save whatever progress we have
        save_index()
        save_metadata()

def process_single_url(url):
    """Process a single URL: scrape, chunk, embed, and store"""
    global all_documents, all_embeddings, all_metadatas, all_ids
    
    try:
        logging.info(f"Processing URL: {url}")
        
        # Delete existing chunks for this URL if any
        check_and_delete_chunks(url)
        
        # Step 1: Scrape the URL
        data = web_scrape_url(url)
        
        if not data:
            logging.warning(f"No data found for {url}. Skipping...")
            return
        
        # Step 2: Process chunks
        scraped_text = str(data)  # Convert to string to ensure compatibility
        chunks = chunk_text(scraped_text)
        
        if not chunks:
            logging.warning(f"No chunks created for {url}. Skipping...")
            return
        
        # Track chunk IDs for this URL
        url_chunks = []
        
        # Step 3: Embed and store each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{url}_chunk_{i+1}"
            url_chunks.append(chunk_id)
            
            # Generate embedding using Ollama's Nomic Embeddings
            try:
                embedding_result = ollama.embeddings(model="nomic-embed-text:latest", prompt=chunk)
                embedding = embedding_result["embedding"]
                
                if not embedding:
                    logging.error(f"Failed to generate embedding for chunk {chunk_id}")
                    continue
                
                # Store the data
                all_documents.append(chunk)
                all_embeddings.append(embedding)
                all_metadatas.append({
                    "source_url": url,
                    "chunk_size": len(chunk),
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                })
                all_ids.append(chunk_id)
                
                logging.info(f"Chunk {chunk_id} processed successfully.")
            
            except Exception as e:
                logging.error(f"Error embedding chunk {i} from {url}: {e}")
                continue
        
        # Update the URL to chunks mapping
        url_to_chunks[url] = url_chunks
        save_metadata()
        
        logging.info(f"Successfully processed URL: {url} with {len(url_chunks)} chunks")
    
    except Exception as e:
        logging.error(f"Error processing URL {url}: {e}")

def main():
    # First try to load existing index
    load_index()
    
    main_url = "https://illinois.edu/sitemap.xml"  # Replace with your main URL or sitemap URL
    if 'sitemap.xml' in main_url:
        urls = fetch_urls_from_sitemap(main_url)
    elif main_url == "":
        logging.error("Invalid URL entered.")
        return
    else:
        urls = [main_url]  # If it's a direct URL instead of a sitemap
    
    logging.info(f"Memory Usage Before Processing: {psutil.virtual_memory().percent}%")
    
    try:
        # Process URLs in batches to manage memory
        process_urls(urls, batch_size=10)
    except Exception as e:
        logging.error(f"Kernel Crash Debug: {e}")
    
    logging.info(f"Memory Usage After Processing: {psutil.virtual_memory().percent}%")
    
    # Save the final index
    save_index()
    save_metadata()

if __name__ == "__main__":
    # Import faiss here to avoid import issues
    
    main()