"""OpenAI API client for document similarity scoring with intelligent rate limiting."""

import json
import time
import re
from typing import Dict, Optional
from openai import OpenAI
import config
from smart_rate_limiter import SmartRateLimiter

class SimilarityScorer:
    """Client for scoring document similarities using GPT-4o-mini."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.API_KEY)
        self.rate_limiter = SmartRateLimiter(config.API_CALLS_PER_SECOND)
        self.model = config.MODEL
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS
        
        # Load anchor documents once
        self.anchors = self._load_anchors()
        
        # Statistics
        self.total_calls = 0
        self.failed_calls = 0
    
    def _load_anchors(self) -> Dict:
        """Load anchor documents from saved file."""
        import json
        with open(config.ANCHOR_PATH, 'r', encoding='utf-8') as f:
            anchor_data = json.load(f)
        
        # Organize anchors by dimension and category
        anchors = {}
        for dimension, dim_data in anchor_data['anchors'].items():
            if 'categories' in dim_data:
                # New format with categories nested
                anchors[dimension] = {}
                for category, cat_data in dim_data['categories'].items():
                    if cat_data and 'document' in cat_data:
                        doc_data = cat_data['document']
                        # Truncate content to first 500 chars for prompt
                        content = doc_data.get('content', '')[:500] + '...'
                        anchors[dimension][category] = {
                            'title': doc_data.get('title', ''),
                            'content': content,
                            'doc_id': doc_data.get('doc_id', '')
                        }
            else:
                # Old format compatibility (direct categories)
                anchors[dimension] = {}
                for category, doc_data in dim_data.items():
                    if isinstance(doc_data, dict) and 'content' in doc_data:
                        # Direct document data
                        content = doc_data.get('content', '')[:500] + '...'
                        anchors[dimension][category] = {
                            'title': doc_data.get('title', ''),
                            'content': content,
                            'doc_id': doc_data.get('doc_id', '')
                        }
        return anchors
    
    def get_document_similarities(self, document: Dict) -> Optional[Dict]:
        """
        Get similarity scores for a document against all 12 anchors in one API call.
        
        Args:
            document: Document dictionary with title and content
            
        Returns:
            Dictionary with similarity scores for all dimensions and categories
        """
        prompt = self._build_similarity_prompt(document)
        
        for attempt in range(config.MAX_RETRY_ATTEMPTS):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert at analyzing carbon trading policy documents. Compare documents based on their semantic meaning and policy implications, not superficial characteristics."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )
                
                # Mark success for rate limiter
                self.rate_limiter.success()
                self.total_calls += 1
                
                # Parse response
                result = json.loads(response.choices[0].message.content)
                
                # Validate structure
                if self._validate_similarity_response(result):
                    return result
                else:
                    print(f"    Invalid response structure for doc {document.get('doc_id', 'unknown')}")
                    if attempt < config.MAX_RETRY_ATTEMPTS - 1:
                        continue
                    return None
                    
            except Exception as e:
                error_str = str(e)
                
                # Check for rate limit error
                if "rate_limit" in error_str.lower():
                    # Extract retry_after if available
                    retry_match = re.search(r'try again in (\d+(?:\.\d+)?)([ms])', error_str)
                    if retry_match:
                        retry_time = float(retry_match.group(1))
                        if retry_match.group(2) == 'ms':
                            retry_time /= 1000
                    else:
                        retry_time = config.EXPONENTIAL_BACKOFF_BASE ** attempt
                    
                    self.rate_limiter.hit_rate_limit(retry_time)
                    
                    if attempt < config.MAX_RETRY_ATTEMPTS - 1:
                        continue
                    
                # Other errors with exponential backoff
                elif attempt < config.MAX_RETRY_ATTEMPTS - 1:
                    wait_time = config.EXPONENTIAL_BACKOFF_BASE ** attempt
                    print(f"    Error: {error_str[:100]}... Retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    print(f"    Failed after {attempt + 1} attempts for doc {document.get('doc_id', 'unknown')}: {error_str[:200]}")
                    self.failed_calls += 1
                    return None
        
        return None
    
    def _build_similarity_prompt(self, document: Dict) -> str:
        """Build the prompt for similarity scoring."""
        # Truncate document content to manage token usage
        doc_content = document.get('content', '')[:1500] + '...'
        
        # Format anchor descriptions
        format_args = {
            'doc_title': document.get('title', ''),
            'doc_content': doc_content
        }
        
        # Add each anchor
        for dimension in ['supply', 'demand', 'policy_strength']:
            for category in self.anchors.get(dimension, {}):
                anchor = self.anchors[dimension][category]
                key = f"{dimension}_{category}" if dimension != 'policy_strength' else f"policy_{category}"
                format_args[key] = f"{anchor['title'][:100]}: {anchor['content'][:200]}"
        
        return config.SIMILARITY_PROMPT_TEMPLATE.format(**format_args)
    
    def _validate_similarity_response(self, response: Dict) -> bool:
        """Validate the structure of the similarity response."""
        required_dimensions = ['supply', 'demand', 'policy_strength']
        
        for dim in required_dimensions:
            if dim not in response:
                return False
            
            # Check categories based on dimension
            if dim == 'supply' or dim == 'demand':
                required_cats = ['major_decrease', 'minor_decrease', 'minor_increase', 'major_increase']
            else:  # policy_strength
                required_cats = ['informational', 'encouraged', 'binding', 'mandatory']
            
            for cat in required_cats:
                if cat not in response[dim]:
                    return False
                # Check if it's a valid number
                try:
                    score = float(response[dim][cat])
                    if not 0 <= score <= 100:
                        return False
                except (TypeError, ValueError):
                    return False
        
        return True
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        stats = self.rate_limiter.get_stats()
        stats.update({
            'total_api_calls': self.total_calls,
            'failed_api_calls': self.failed_calls,
            'success_rate': (self.total_calls - self.failed_calls) / max(self.total_calls, 1)
        })
        return stats