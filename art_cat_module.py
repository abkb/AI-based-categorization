import torch
from sentence_transformers import SentenceTransformer, util
import openpyxl
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from pathlib import Path
import os
import sys
import re

# ==============================================================================
# 0. LIBRARY CHECKS & IMPORTS
# ==============================================================================

try:
    import fitz # PyMuPDF for PDF reading
except ImportError:
    print("Error: The 'PyMuPDF' library (fitz) is required. Please install it using 'pip install pymupdf'.", file=sys.stderr)
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("Error: The 'sentence-transformers' library is not installed. Please install it.", file=sys.stderr)
    sys.exit(1)

try:
    from pyzotero import zotero
except ImportError:
    print("Error: The 'pyzotero' library is not installed. Please install it.", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# 1. CONFIGURATION CONSTANTS (FILL THESE OUT)
# ==============================================================================

# Zotero Credentials (API key still needed for metadata, but not file download)
ZOTERO_USER_ID = "8427826" # <-- REPLACE
ZOTERO_API_KEY = "FzvL5N9EfLMhNMjKO65hJTTx" # <-- REPLACE
ZOTERO_COLLECTION_KEY = "7IMN8GN2"  # <-- REPLACE with collection key (e.g., "7IMN8GN2") or None for entire library

# 🛑 CRITICAL NEW CONFIGURATION: LOCAL ZOTERO STORAGE PATH
# You MUST replace this with the absolute path to your Zotero Data Directory.
# This directory contains the 'storage' folder.
# Find this path in Zotero Preferences -> Advanced -> Files and Folders -> Data Directory Location.
# Examples:
# Windows: "C:\\Users\\YourName\\Zotero"
# Mac: "/Users/YourName/Zotero"
# Linux: "/home/YourName/Zotero"
ZOTERO_LOCAL_STORAGE_PATH = "/Users/akibria/Zotero/storage" # <--- REPLACE THIS LINE!

# AI Model and Parameters
MODEL_NAME = 'all-mpnet-base-v2'
MIN_CONFIDENCE = 0.35
BATCH_SIZE = 16 # Adjust based on GPU memory
TOP_N = 3

# Theme Definitions (Remaining the same)
THEMES = [
    "Climate Adaptation", "Climate Mitigation", "Agricultural Production",
    "Carbon Emissions & Sequestration", "Biodiversity Conservation",
    "Renewable Energy", "Water Resource Management", "Soil Health & Management",
    "Environmental Policy", "Sustainable Development", "Land Use Change",
    "Forest Management", "Environmental Technologies", "Wetland Restoration"
]

# Domain Validation Keywords (Remaining the same)
DOMAIN_KEYWORDS = {
    # ... (content of DOMAIN_KEYWORDS remains the same)
    "Soil Health & Management": {
        "required": ["soil", "compost", "organic matter", "tilth", "fertility", "erosion", "tillage"],
        "exclusion": ["semiconductor", "quantum", "crystal", "epitaxial", "TEM", "STEM", "dislocation"]
    },
    "Agricultural Production": {
        "required": ["crop", "yield", "harvest", "agriculture", "farming", "irrigation", "pesticide"],
        "exclusion": ["semiconductor", "quantum", "electron", "microscopy", "nanowire"]
    },
    "Climate Adaptation": {
        "required": ["adaptation", "adaptive", "adapting", "climate adaptation"],
        "exclusion": []
    },
    "Climate Mitigation": {
        "required": ["mitigation", "mitigating", "mitigative", "climate mitigation", "GHG emissions", "greenhouse"],
        "exclusion": []
    },
    "Carbon Emissions & Sequestration": {
        "required": ["carbon", "emission", "sequestration", "carbon sequestration", "greenhouse gas"],
        "exclusion": []
    },
    "Biodiversity Conservation": {
        "required": ["biodiversity", "species", "ecosystem", "conservation"],
        "exclusion": []
    },
    "Renewable Energy": {
        "required": ["energy", "renewable", "solar", "wind", "geothermal"],
        "exclusion": []
    },
    "Water Resource Management": {
        "required": ["water", "hydro", "aquatic", "water resource"],
        "exclusion": []
    },
    "Sustainable Development": {
        "required": ["sustainability", "sustainable"],
        "exclusion": []
    },
    "Land Use Change": {
        "required": ["land use", "land cover", "deforestation"],
        "exclusion": []
    },
    "Forest Management": {
        "required": ["forest", "forestry", "reforestation", "afforestation"],
        "exclusion": []
    },
    "Wetland Restoration": {
        "required": ["wetland", "restoration", "ecosystem restoration", "peatland"],
        "exclusion": []
    }
}


# ==============================================================================
# 2. DATA STRUCTURES (FIXED: Moved up to avoid NameError)
# ==============================================================================

@dataclass
class Article:
    """Represents a single article with its metadata and content."""
    author: str
    year: int
    title: str
    content: str
    themes: str = "N/A"
    top_scores: List[Tuple[str, float]] = field(default_factory=list)

# ==============================================================================
# 3. CONTENT CLEANING (Remaining the same)
# ==============================================================================

def clean_article_content_for_keywords(content: str) -> str:
    """
    Cleans text content by attempting to remove common trailing sections (e.g., References)
    that are irrelevant for core theme identification.
    """
    if not content:
        return ""
        
    content_lower = content.lower()
    
    # Common section headers to mark the end of the main body
    termination_patterns = [
        r'acknowledgements?',
        r'references?',
        r'appendi(x|ces?)',
        r'author contributions?',
        r'funding',
        r'bibliograph(y|ie)'
    ]
    
    # Find the earliest termination point
    min_index = len(content)
    for pattern in termination_patterns:
        match = re.search(pattern, content_lower)
        if match:
            # Check if the match is near the end, and choose the earliest point
            match_index = match.start()
            if match_index > 500: # Only truncate if a substantial body exists before the match
                min_index = min(min_index, match_index)
                
    cleaned_content = content[:min_index]
    
    # Simple cleaning (removes multiple newlines, standardizes spaces)
    cleaned_content = re.sub(r'\n{2,}', '\n', cleaned_content)
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
    
    return cleaned_content

# ==============================================================================
# 4. ZOTERO INTEGRATION (MODIFIED FOR LOCAL ACCESS)
# ==============================================================================

def extract_text_from_pdf_data(pdf_data: bytes) -> str:
    """Uses PyMuPDF (fitz) to extract text from raw PDF bytes."""
    text = ""
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        # print(f"Warning: Failed to parse PDF data. Error: {e}", file=sys.stderr)
        return ""
    return text.strip()

def fetch_zotero_articles(user_id: str, api_key: str, collection_key: str = None) -> List[Article]:
    """
    Fetches article metadata from Zotero API and PDF content from the local disk.
    """
    try:
        # We still need pyzotero to fetch item metadata and child attachment keys
        zot = zotero.Zotero(user_id, 'user', api_key)
        
        print("Connecting to Zotero for metadata...")
        
        item_types_list = ['journalArticle', 'magazineArticle', 'report']
        
        fetch_kwargs = {
            'format': 'json',
            'children': 1,
            'limit': 100
        }
        
        if collection_key:
            print(f"Fetching articles from collection '{collection_key}'...")
            items = zot.collection_items(collection_key, **fetch_kwargs)
        else:
            print("Fetching all top-level articles...")
            items = zot.top_level_items(**fetch_kwargs)

        articles_list = []
        for item in items:
            if item['data']['itemType'] in item_types_list: 
                
                title = item['data'].get('title', 'N/A')
                # Start with abstract as fallback content
                content = item['data'].get('abstractNote', item['data'].get('title', ''))
                
                # --- PDF Full Text Extraction Logic (LOCAL READ) ---
                try:
                    # Explicitly fetch children to get full attachment metadata (including filename/path hash)
                    attachments = zot.children(item['key'])
                except Exception as e:
                    print(f"   ⚠️ Could not fetch children metadata for {title}: {e}")
                    attachments = []
                
                pdf_attachment = next((a for a in attachments if a['data'].get('contentType') == 'application/pdf'), None)

                if pdf_attachment:
                    try:
                        print(f"  -> Attempting local read for: {title[:60]}...")
                        
                        # 1. Get the necessary keys from the attachment metadata
                        attachment_key = pdf_attachment['data']['key']
                        attachment_filename = pdf_attachment['data']['filename']
                        
                        # 2. Construct the local path: [Local_Path]/storage/[key]/[filename]
                        local_pdf_path = Path(ZOTERO_LOCAL_STORAGE_PATH) / 'storage' / attachment_key / attachment_filename
                        
                        if local_pdf_path.exists():
                            # 3. Read the raw bytes directly from the local disk
                            pdf_data = local_pdf_path.read_bytes() 
                            
                            full_text = extract_text_from_pdf_data(pdf_data)
                            
                            if full_text and len(full_text) > 500:
                                content = full_text # Overwrite abstract with full text
                                print("     ✅ Full text extracted from local disk.")
                            else:
                                print("     ⚠️ PDF parsing failed or text too short. Falling back to Abstract.")
                        else:
                            print(f"     ❌ File not found at local path: {local_pdf_path}")
                            
                    except Exception as e:
                        print(f"     ❌ Error reading or parsing local PDF: {e}. Falling back to Abstract.")
                else:
                    print(f"     ⚠️ No PDF attachment found for {title}.")
                
                # 5. Apply content cleaning to the final content source (full text or abstract)
                cleaned_content = clean_article_content_for_keywords(content)
                        
                authors = ", ".join([f"{c.get('lastName', '')}, {c.get('firstName', '')}" for c in item['data'].get('creators', [])])
                
                year = 0
                date_str = item['data'].get('date', '')
                if date_str:
                    try:
                        year = int(date_str[:4])
                    except ValueError:
                        pass
                        
                articles_list.append(Article(
                    author=authors,
                    year=year,
                    title=title,
                    content=cleaned_content # Use the cleaned content for analysis
                ))
        
        print(f"\nSuccessfully fetched and processed {len(articles_list)} articles.")
        return articles_list
        
    except Exception as e:
        print(f"Fatal Error during Zotero processing: {e}", file=sys.stderr)
        return []


# ==============================================================================
# 5. DOMAIN VALIDATION FUNCTIONS (Remaining the same)
# ==============================================================================

def contains_keywords(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the specified keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)

def validate_theme_relevance(article: Article, theme: str, score: float) -> bool:
    """Validate whether a theme is actually relevant using keyword validation."""
    if theme not in DOMAIN_KEYWORDS:
        return True
        
    rules = DOMAIN_KEYWORDS[theme]
    # Check against the cleaned content
    full_text = f"{article.title} {article.content}".lower()
    
    # Check for exclusion keywords first
    if rules.get("exclusion") and contains_keywords(full_text, rules["exclusion"]):
        return False
        
    # Check for required keywords (if specified)
    if rules.get("required") and not contains_keywords(full_text, rules["required"]):
        return False
        
    return True

def filter_irrelevant_themes(article: Article, theme_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Filter out themes that are semantically similar but contextually irrelevant."""
    filtered_scores = []
    for theme, score in theme_scores:
        if validate_theme_relevance(article, theme, score):
            filtered_scores.append((theme, score))
            
    return filtered_scores

# ==============================================================================
# 6. AI AGENT CLASS (Remaining the same)
# ==============================================================================

class ArticleCategorizer:
    """Categorizes articles by semantic similarity to predefined themes."""
    def __init__(self, themes: List[str], model_name: str, min_confidence: float = 0.35):
        """Initializes the categorizer by loading the model and themes."""
        self.themes = themes
        self.min_confidence = min_confidence
        self.model = self._load_model(model_name)
        if self.model:
            # Handle case where GPU is available
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.theme_embeddings = self.model.encode(self.themes, convert_to_tensor=True, device=self.device)
            print(f"Categorizer ready. Using device: {self.device}")

    def _load_model(self, model_name: str):
        """Loads the SentenceTransformer model with error handling."""
        try:
            model = SentenceTransformer(model_name)
            print(f"Model '{model_name}' loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}", file=sys.stderr)
            print("Please ensure you have an internet connection or the model is cached locally.", file=sys.stderr)
            return None

    def categorize(self, articles: List[Article]) -> List[Article]:
        """Categorizes a list of articles based on theme similarity using batch processing."""
        if not self.model:
            print("Categorization skipped due to model loading failure.")
            return articles

        article_contents = [a.content for a in articles if a.content]
        valid_articles = [a for a in articles if a.content]
        
        if not article_contents:
            print("No content available for categorization.")
            return articles

        article_embeddings_list = []
        for i in range(0, len(article_contents), BATCH_SIZE):
            batch = article_contents[i:i + BATCH_SIZE]
            print(f"Processing batch {i // BATCH_SIZE + 1}...")
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True, device=self.device)
            article_embeddings_list.append(batch_embeddings)
            
        article_embeddings = torch.cat(article_embeddings_list)

        categorized_articles: List[Article] = []
        for i, article in enumerate(valid_articles):
            cosine_scores = util.cos_sim(article_embeddings[i], self.theme_embeddings)[0]

            scores = [(self.themes[j], float(score)) for j, score in enumerate(cosine_scores)]
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            # FILTER OUT IRRELEVANT THEMES BASED ON DOMAIN KNOWLEDGE
            filtered_scores = filter_irrelevant_themes(article, scores)
            
            # Select themes above threshold from filtered list
            selected_themes = [theme for theme, score in filtered_scores if score >= self.min_confidence]
            
            # Fallback logic
            if not selected_themes:
                 if filtered_scores: # Fallback to top filtered theme
                     selected_themes = [filtered_scores[0][0]]
                 elif scores: # Final fallback to original top theme (if filtering removed everything)
                      selected_themes = [scores[0][0]]
                 else:
                      selected_themes = ["Uncategorized"]

            # Update the article object with results
            article.themes = ", ".join(selected_themes)
            # Store only the top N scores from the filtered list
            article.top_scores = filtered_scores[:TOP_N]
            categorized_articles.append(article)

        return categorized_articles

    def save_to_excel(self, articles: List[Article], filename: str):
        """Saves categorized articles into an Excel file."""
        try:
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "Categorized Articles"

            headers = ["Author", "Title", "Year", "Themes", "Content (Filtered)"] + THEMES
            sheet.append(headers)

            for article in articles:
                row_data = [
                    article.author,
                    article.title,
                    article.year,
                    article.themes,
                    article.content,
                ]

                top_scores_dict = {theme: score for theme, score in article.top_scores}

                for theme in THEMES:
                    score = top_scores_dict.get(theme)
                    if score is not None:
                        # Append the percentage number
                        row_data.append(round(score * 100, 2))
                    else:
                        # Append the exclusion marker
                        row_data.append("-")

                sheet.append(row_data)

            output_path = Path(filename)
            workbook.save(output_path)
            print(f"Results saved to '{output_path}'")
        except Exception as e:
            print(f"Error saving Excel file '{filename}': {e}", file=sys.stderr)
            
# ==============================================================================
# 7. ORCHESTRATOR ENTRY POINT (Remaining the same)
# ==============================================================================

def run_categorization_and_save(output_filename: str = "zotero_analysis_output.xlsx") -> bool:
    """
    Core function to run the categorization pipeline.
    """
    print("--- Starting Zotero Categorization Pipeline ---")
    
    # 🛑 CRITICAL CHECK
    if ZOTERO_LOCAL_STORAGE_PATH == "/path/to/your/Zotero/Data/Folder":
        print("\nFATAL ERROR: You must replace ZOTERO_LOCAL_STORAGE_PATH with your actual Zotero data path to enable local file reading.", file=sys.stderr)
        return False
        
    try:
        # Fetch articles from Zotero using module-level configurations
        articles_from_zotero = fetch_zotero_articles(
            user_id=ZOTERO_USER_ID, 
            api_key=ZOTERO_API_KEY,
            collection_key=ZOTERO_COLLECTION_KEY
        )
        
        if not articles_from_zotero:
            print("Categorization Agent: No articles fetched. Cannot proceed.")
            return False

        categorizer = ArticleCategorizer(
            themes=THEMES, 
            model_name=MODEL_NAME, 
            min_confidence=MIN_CONFIDENCE
        )
        
        if categorizer.model:
            results = categorizer.categorize(articles_from_zotero)
            categorizer.save_to_excel(results, output_filename)
            return True
        
        print("Categorization failed due to model initialization error.")
        return False
            
    except Exception as e:
        print(f"Categorization Agent failed during run: {e}")
        return False
        
# ========================================================================
####### TEST ########
if __name__ == '__main__':
    
    if ZOTERO_API_KEY == "FzvL5N9EfLMhNMjKO65hJTTx" and ZOTERO_LOCAL_STORAGE_PATH == "/path/to/your/Zotero/Data/Folder":
         print("\n*** WARNING: Zotero credentials are NOT set. Running a quick MOCK test. ***")
         
         # --- MOCK TEST SETUP ---
         mock_article = Article(
             author="Test Author",
             year=2024,
             title="Tillage and Carbon Sequestration Policy",
             content=clean_article_content_for_keywords(
                 "This study investigates the effect of minimum tillage and cover crops "
                 "on soil organic matter and soil fertility. We analyze the carbon sequestration "
                 "potential of these agricultural production methods as a climate mitigation strategy. "
                 "This paper excludes all research on nuclear power. REFERENCES. This is the end."
             )
         )
         
         categorizer = ArticleCategorizer(themes=THEMES, model_name=MODEL_NAME, min_confidence=0.35)
         
         if categorizer.model:
             results = categorizer.categorize([mock_article]) 
             test_result = results[0]
             
             print("\n--- ✅ Mock Analysis Results ---")
             print(f"Title: {test_result.title}")
             print(f"Themes: \033[92m{test_result.themes}\033[0m")
             print("Top Scores (Filtered):")
             for theme, score in test_result.top_scores:
                 print(f" - {theme}: {score:.4f}")
             print("\n--- End of Mock Test ---")

         else:
             print("Mock test failed: Could not load the SentenceTransformer model.")
             
    else:
        # This will run the full Zotero process
        run_categorization_and_save()