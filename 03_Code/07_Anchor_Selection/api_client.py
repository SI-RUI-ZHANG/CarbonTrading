"""OpenAI API client for document classification and comparison."""

import json
import time
import re
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
import config
from smart_rate_limiter import SmartRateLimiter

class DocumentClassifier:
    """Client for classifying and comparing documents using GPT-4o-mini."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.API_KEY)
        self.rate_limiter = SmartRateLimiter(config.API_CALLS_PER_SECOND)
        self.model = config.MODEL
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS
        
    def classify_document(self, document: Dict) -> List[Tuple[str, str]]:
        """Classify a document across all dimensions with rate limiting.
        
        Args:
            document: Document dictionary with content
            
        Returns:
            List of (dimension, category) tuples for relevant classifications
        """
        prompt = self._build_classification_prompt(document)
        
        for attempt in range(config.MAX_RETRY_ATTEMPTS):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的碳市场政策分析师。请仔细分析政策文档并准确分类。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )
                
                # Mark success for rate limiter
                self.rate_limiter.success()
                
                result = json.loads(response.choices[0].message.content)
                classifications = []
                
                if 'classifications' in result:
                    for item in result['classifications']:
                        if 'dimension' in item and 'category' in item:
                            # Validate dimension and category
                            dim = item['dimension']
                            cat = item['category']
                            if dim in config.DIMENSIONS:
                                # Check if category is valid for this dimension
                                if dim in config.DIMENSION_CATEGORIES:
                                    if cat in config.DIMENSION_CATEGORIES[dim]:
                                        classifications.append((dim, cat))
                                # Fallback for backward compatibility
                                elif cat in config.CATEGORIES:
                                    classifications.append((dim, cat))
                
                return classifications
                
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
                    print(f"    Error: {error_str[:100]}... Retrying in {wait_time}s (attempt {attempt + 1}/{config.MAX_RETRY_ATTEMPTS})")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    print(f"    Failed after {attempt + 1} attempts for doc {document.get('doc_id', 'unknown')}: {error_str[:200]}")
                    return []
    
    def compare_documents(self, doc_a: Dict, doc_b: Dict, dimension: str, category: str) -> str:
        """Compare two documents to determine which better represents a dimension-category.
        
        Args:
            doc_a: First document
            doc_b: Second document
            dimension: Dimension to compare for
            category: Category to compare for
            
        Returns:
            'A' if doc_a is better, 'B' if doc_b is better
        """
        prompt = self._build_comparison_prompt(doc_a, doc_b, dimension, category)
        
        for attempt in range(config.MAX_RETRY_ATTEMPTS):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的碳市场政策分析师。请基于内容的代表性和清晰度选择更好的文档。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Lower temperature for more consistent comparison
                    max_tokens=100
                )
                
                # Mark success for rate limiter
                self.rate_limiter.success()
                
                result = response.choices[0].message.content.strip().upper()
                
                # Extract A or B from response
                if 'A' in result and 'B' not in result:
                    return 'A'
                elif 'B' in result and 'A' not in result:
                    return 'B'
                else:
                    # Default to A if unclear
                    return 'A'
                    
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
                    print(f"    Error comparing: {error_str[:100]}... Retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    print(f"    Failed comparing after {attempt + 1} attempts: {error_str[:200]}")
                    return 'A'  # Default to keeping current
    
    def _build_classification_prompt(self, document: Dict) -> str:
        """Build the classification prompt for a document."""
        
        content = document.get('content', '')
        title = document.get('title', '')
        
        prompt = f"""分析以下碳市场政策文档，判断它与各维度的相关性并分类。

文档标题：{title}

文档内容：
{content}

请分析该文档在以下维度的归类：

维度1：配额供给 - 是否影响碳配额的供给量？
关注：配额总量设定、配额分配方法、配额发放规则、储备配额释放、拍卖数量等
如相关，归类为：
- major_decrease：大幅减少供给（如总量削减>10%、暂停配额发放）
- minor_decrease：适度减少供给（如总量削减5-10%、收紧分配基准）  
- minor_increase：适度增加供给（如释放储备配额、增加拍卖量）
- major_increase：大幅增加供给（如总量增加>10%、特别配额发放）

维度2：配额需求 - 是否影响碳配额的需求量？
关注：控排企业范围、履约标准、抵消机制、处罚力度、行业扩容等
如相关，归类为：
- major_increase：大幅增加需求（如新增行业纳入、提高履约标准>10%）
- minor_increase：适度增加需求（如企业范围微调、履约标准提高5-10%）
- minor_decrease：适度减少需求（如增加抵消比例、放宽履约要求）
- major_decrease：大幅减少需求（如行业退出、大幅降低履约标准）

维度3：政策强度 - 政策的约束力和执行强度如何？
关注：法规层级、强制性表述、处罚条款、执行要求等
归类为：
- mandatory：强制执行（含"必须"、"应当"、"不得"、明确处罚）
- binding：约束指导（含"应"、"原则上"、"一般要求"）
- encouraged：鼓励支持（含"鼓励"、"支持"、"可以"、"引导"）
- informational：信息公告（仅通知、公示、解读、培训等）

返回JSON格式（每个维度最多一个类别）：
{{
  "classifications": [
    {{"dimension": "supply", "category": "minor_decrease"}},
    {{"dimension": "policy_strength", "category": "mandatory"}},
    ...
  ]
}}

如果文档与所有维度都不相关，返回空列表：
{{
  "classifications": []
}}"""
        
        return prompt
    
    def _build_comparison_prompt(self, doc_a: Dict, doc_b: Dict, dimension: str, category: str) -> str:
        """Build the comparison prompt for two documents."""
        
        dim_info = config.DIMENSIONS[dimension]
        # Get category description from the dimension-specific categories
        if dimension in config.DIMENSION_CATEGORIES:
            cat_desc = config.DIMENSION_CATEGORIES[dimension].get(category, category)
        else:
            cat_desc = config.CATEGORIES.get(category, category)
        
        prompt = f"""比较两份文档，选择哪份是【{dim_info['name']}】维度【{cat_desc}】类别的更好代表。

维度说明：{dim_info['description']}
类别说明：{cat_desc}

文档A标题：{doc_a.get('title', '')}
文档A内容：
{doc_a.get('content', '')}

---

文档B标题：{doc_b.get('title', '')}
文档B内容：
{doc_b.get('content', '')}

请选择哪份文档更清晰、更具代表性地体现了这个维度和类别。
只需回答：A 或 B"""
        
        return prompt


if __name__ == "__main__":
    # Test the classifier with rate limiting
    classifier = DocumentClassifier()
    
    # Test document
    test_doc = {
        'doc_id': 'test_001',
        'title': '关于调整2025年度碳排放配额分配方案的通知',
        'content': '为进一步完善碳排放权交易市场，经研究决定，2025年度碳排放配额总量较2024年度减少15%，重点控排企业配额分配采用基准线法。'
    }
    
    print("Testing classification with rate limiting...")
    print(f"Rate limit: {config.API_CALLS_PER_SECOND} calls/second")
    
    classifications = classifier.classify_document(test_doc)
    print(f"Classifications: {classifications}")
    
    # Print rate limiter stats
    stats = classifier.rate_limiter.get_stats()
    print(f"\nRate limiter stats:")
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Rate limit hits: {stats['total_rate_limit_hits']}")
    print(f"  Total wait time: {stats['total_wait_time']:.2f}s")