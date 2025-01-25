import os
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import ollama
from chromadb.config import Settings
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import chromadb
import json

from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

# Initialize ChromaDB client with embedding function
embedding_function = OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
)

# Persistent ChromaDB client setup
client = chromadb.PersistentClient(path="./demo-rag-chroma" ,settings=Settings(anonymized_telemetry=False))

# Create a collection (if not exists) in ChromaDB
def get_collection():
    """
    Create or retrieve a ChromaDB collection with the Ollama embedding function.
    """
    return client.get_or_create_collection(
        name="site_pages_collection",
        embedding_function=embedding_function
    )



@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama."""
    try:
        # Use Ollama's embedding generation
        response = ollama.embeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 768  # Adjust vector size based on your embedding model
    

async def insert_chunk(chunk: ProcessedChunk):
    """
    Insert the processed chunk into ChromaDB using the updated embedding function.
    """
    try:
        collection = get_collection()

        # Prepare data for insertion into ChromaDB
        chunk_id = f"{chunk.url}_{chunk.chunk_number}"  # Unique identifier for the chunk
        collection.add(
            documents=[chunk.content],  # Original text
            metadatas=[chunk.metadata],  # Metadata
            embeddings=[chunk.embedding],  # Embedding vector
            ids=[chunk_id]  # Unique IDs
        )
        
        print(f"Inserted/Updated chunk {chunk.chunk_number} for {chunk.url} into ChromaDB")
    except Exception as e:
        print(f"ChromaDB Insertion Error: {e}")

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using Ollama."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        # Use Ollama's generate method
        response = await asyncio.to_thread(
            ollama.chat,
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ],
            format="json"
        )
        print(json.loads(response['message']['content']))
        return json.loads(response['message']['content'])
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )


async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)



async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()



def fetch_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """
    Recursively fetches all URLs from a sitemap or nested sitemaps.

    Args:
        sitemap_url (str): The root or nested sitemap URL.

    Returns:
        List[str]: List of all URLs to be crawled.
    """
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
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
    except Exception as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
        return set()




async def main():
    # Get URLs from Pydantic AI docs
    siteurl="https://illinois.edu/sitemap.xml"
    urls = list(fetch_urls_from_sitemap(siteurl))
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
