"""
Helper function for fetching URL content in Qwen CLI.
This function will be dynamically loaded by the CLI.
"""

import logging
import urllib.request
import urllib.error
import re

logger = logging.getLogger("qwen_cli.helpers.fetch_url")

def handle_fetch_url(url: str):
    """
    Whenever the user asks to read, load, research or consult a URL,
    write this command: `[FETCH_URL url]` (e.g., `[FETCH_URL https://example.com/info.txt]`).

    This will fetch the content from the provided URL and return it formatted for the model context.
    """
    logger.info(f"Fetching URL: {url}")
    
    url = url.strip('"').strip("'")
    if not re.match(r'^https?://', url):
        logger.error(f"Invalid URL format: {url}")
        return None
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 Qwen CLI URL Fetcher',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        request = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(request, timeout=10) as response:
            content_type = response.info().get_content_type()
            
            accepted_content_types = ['text', 'json', 'xml']
            if any(content_type.startswith(ct) for ct in accepted_content_types):
                content = response.read().decode('utf-8', errors='replace')
                
                if len(content) > 100000:
                    content = content[:100000] + "\n\n[Content truncated due to size]"
                
                formatted_content = f"[URL: {url}]\n```\n{content}\n```"
                
                logger.info(f"Successfully fetched URL: {url} ({len(content)} characters)")
                return formatted_content
            else:
                logger.warning(f"Unsupported content type: {content_type}")
                return f"[URL: {url}]\nUnsupported content type: {content_type}. Only text-based content can be fetched."
                
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error for {url}: {e.code} {e.reason}")
        return f"[URL: {url}]\nHTTP Error: {e.code} {e.reason}"
        
    except urllib.error.URLError as e:
        logger.error(f"URL Error for {url}: {e.reason}")
        return f"[URL: {url}]\nURL Error: {e.reason}"
        
    except TimeoutError:
        logger.error(f"Timeout fetching {url}")
        return f"[URL: {url}]\nTimeout: The request timed out after 10 seconds"
        
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None