"""OpenAI API client for document similarity scoring with intelligent rate limiting."""

import json
import time
import re
from typing import Dict, Optional
from openai import OpenAI
import config
from smart_rate_limiter import SmartRateLimiter

class DocumentPositioner:
    """Client for positioning documents on policy spectrums using GPT-4o."""
    
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
    
    def get_document_positions(self, document: Dict) -> Optional[Dict]:
        """
        Get spectrum positions for a document by comparing to reference anchors.
        
        Args:
            document: Document dictionary with title and content
            
        Returns:
            Dictionary with positions on each spectrum
        """
        prompt = self._build_position_prompt(document)
        
        for attempt in range(config.MAX_RETRY_ATTEMPTS):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert at positioning carbon trading policy documents on intensity spectrums. Determine exact positions based on policy content and implications."
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
                if self._validate_position_response(result):
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
    
    def _build_position_prompt(self, document: Dict) -> str:
        """Build the prompt for position scoring."""
        # Truncate document content to manage token usage
        doc_content = document.get('content', '')[:1500] + '...'
        
        # Format anchor descriptions
        format_args = {
            'doc_title': document.get('title', ''),
            'doc_content': doc_content
        }
        
        # Add each anchor
        for dimension in ['supply', 'demand', 'policy_strength']:
            # Get the correct categories for this dimension
            if dimension == 'policy_strength':
                categories = ['informational', 'encouraged', 'binding', 'mandatory']
            else:
                categories = ['restrict', 'reduce', 'increase', 'expand']
            
            for category in categories:
                if dimension in self.anchors and category in self.anchors[dimension]:
                    anchor = self.anchors[dimension][category]
                    key = f"{dimension}_{category}" if dimension != 'policy_strength' else f"policy_{category}"
                    format_args[key] = f"{anchor['title'][:100]}: {anchor['content'][:200]}"
                else:
                    # Provide empty anchor if missing
                    key = f"{dimension}_{category}" if dimension != 'policy_strength' else f"policy_{category}"
                    format_args[key] = "No anchor document available"
        
        return config.POSITION_SCORING_PROMPT.format(**format_args)
    
    def _validate_position_response(self, response: Dict) -> bool:
        """Validate the structure of the position response."""
        required_fields = ['supply', 'demand', 'policy_strength']
        
        for field in required_fields:
            if field not in response:
                return False
            
            # Check if it's a valid number in the correct range
            try:
                position = float(response[field])
                if field == 'policy_strength':
                    if not 0 <= position <= 150:
                        return False
                else:  # supply or demand
                    if not -150 <= position <= 150:
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