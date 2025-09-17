"""
Evaluation Script for AI Price Comparison Agent
Measures both model-level and agent-level performance metrics
"""

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import List, Dict, Tuple
import time
from tqdm import tqdm
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import logging

# Download required NLTK data
nltk.download('punkt', quiet=True)

# Import your modules
from agent.core import PriceComparisonAgent
from models.inference import ImageToTextModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates the image-to-text model performance
    """
    
    def __init__(self, model_path=None, use_finetuned=False):
        self.model = ImageToTextModel(use_finetuned=use_finetuned)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method4
        
    def calculate_bleu_score(self, reference: str, hypothesis: str) -> float:
        """
        Calculate BLEU score between reference and generated text
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        # Calculate BLEU with smoothing for short sentences
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self.smoothing)
        return score
    
    def calculate_rouge_scores(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def calculate_keyword_accuracy(self, reference: str, hypothesis: str, 
                                  important_keywords: List[str] = None) -> Dict[str, float]:
        """
        Calculate keyword matching accuracy
        """
        ref_lower = reference.lower()
        hyp_lower = hypothesis.lower()
        
        # Default important keywords for fashion
        if important_keywords is None:
            important_keywords = ['brand', 'color', 'type', 'gender', 'material']
        
        # Extract keywords from reference
        ref_words = set(ref_lower.split())
        hyp_words = set(hyp_lower.split())
        
        # Calculate overlap
        overlap = ref_words.intersection(hyp_words)
        
        metrics = {
            'exact_match': float(ref_lower == hyp_lower),
            'word_overlap': len(overlap) / len(ref_words) if ref_words else 0,
            'precision': len(overlap) / len(hyp_words) if hyp_words else 0,
            'recall': len(overlap) / len(ref_words) if ref_words else 0
        }
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0
        
        return metrics
    
    def evaluate_model(self, test_csv_path: str, num_samples: int = None) -> Dict:
        """
        Evaluate model on test dataset
        """
        print("📊 Evaluating Image-to-Text Model...")
        
        # Load test data
        test_df = pd.read_csv(test_csv_path)
        
        if num_samples:
            test_df = test_df.head(num_samples)
        
        results = {
            'bleu_scores': [],
            'rouge1_scores': [],
            'rouge2_scores': [],
            'rougeL_scores': [],
            'keyword_accuracy': [],
            'inference_times': []
        }
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            try:
                # Load image
                image_path = row['image_path']
                image = Image.open(image_path)
                
                # Generate description
                start_time = time.time()
                generated_text = self.model.generate_search_query(image)
                inference_time = time.time() - start_time
                
                # Get reference text
                reference_text = row.get('enhanced_description', row.get('description', ''))
                
                # Calculate metrics
                bleu = self.calculate_bleu_score(reference_text, generated_text)
                rouge = self.calculate_rouge_scores(reference_text, generated_text)
                keyword = self.calculate_keyword_accuracy(reference_text, generated_text)
                
                # Store results
                results['bleu_scores'].append(bleu)
                results['rouge1_scores'].append(rouge['rouge1'])
                results['rouge2_scores'].append(rouge['rouge2'])
                results['rougeL_scores'].append(rouge['rougeL'])
                results['keyword_accuracy'].append(keyword['f1'])
                results['inference_times'].append(inference_time)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {idx}: {e}")
                continue
        
        # Calculate averages
        summary = {
            'avg_bleu': np.mean(results['bleu_scores']),
            'avg_rouge1': np.mean(results['rouge1_scores']),
            'avg_rouge2': np.mean(results['rouge2_scores']),
            'avg_rougeL': np.mean(results['rougeL_scores']),
            'avg_keyword_f1': np.mean(results['keyword_accuracy']),
            'avg_inference_time': np.mean(results['inference_times']),
            'total_samples': len(results['bleu_scores'])
        }
        
        return summary, results

class AgentEvaluator:
    """
    Evaluates end-to-end agent performance
    """
    
    def __init__(self, use_finetuned=False):
        self.agent = PriceComparisonAgent(use_finetuned=use_finetuned)
        
    def create_test_queries(self) -> List[Dict]:
        """
        Create test queries with expected results
        """
        test_queries = [
            {
                'query': 'Nike Air Force 1 white sneakers',
                'expected_keywords': ['nike', 'air', 'force', 'white', 'sneakers'],
                'category': 'footwear'
            },
            {
                'query': 'Adidas running shoes black',
                'expected_keywords': ['adidas', 'running', 'shoes', 'black'],
                'category': 'footwear'
            },
            {
                'query': 'Levi denim jacket blue',
                'expected_keywords': ['levi', 'denim', 'jacket', 'blue'],
                'category': 'apparel'
            },
            {
                'query': 'Samsung Galaxy S23 Ultra',
                'expected_keywords': ['samsung', 'galaxy', 's23', 'ultra'],
                'category': 'electronics'
            },
            {
                'query': 'Apple MacBook Pro 16 inch',
                'expected_keywords': ['apple', 'macbook', 'pro', '16'],
                'category': 'electronics'
            }
        ]
        return test_queries
    
    def evaluate_search_relevance(self, results: List[Dict], query: str, 
                                 expected_keywords: List[str]) -> Dict:
        """
        Evaluate the relevance of search results
        """
        if not results:
            return {
                'has_results': False,
                'top1_relevance': 0,
                'top3_relevance': 0,
                'avg_relevance': 0
            }
        
        relevance_scores = []
        
        for result in results[:5]:
            title_lower = result.get('title', '').lower()
            query_lower = query.lower()
            
            # Check keyword presence
            keyword_matches = sum(1 for kw in expected_keywords if kw in title_lower)
            keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0
            
            # Check query match
            query_words = set(query_lower.split())
            title_words = set(title_lower.split())
            query_match = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
            
            # Combined relevance
            relevance = (keyword_score + query_match) / 2
            relevance_scores.append(relevance)
        
        return {
            'has_results': True,
            'top1_relevance': relevance_scores[0] if relevance_scores else 0,
            'top3_relevance': np.mean(relevance_scores[:3]) if len(relevance_scores) >= 3 else np.mean(relevance_scores),
            'avg_relevance': np.mean(relevance_scores)
        }
    
    def evaluate_price_extraction(self, results: List[Dict]) -> Dict:
        """
        Evaluate price extraction accuracy
        """
        valid_prices = [r for r in results if r.get('price', 0) > 0]
        
        return {
            'price_extraction_rate': len(valid_prices) / len(results) if results else 0,
            'has_prices': len(valid_prices) > 0,
            'num_valid_prices': len(valid_prices),
            'price_range': (min(r['price'] for r in valid_prices), 
                          max(r['price'] for r in valid_prices)) if valid_prices else (0, 0)
        }
    
    def evaluate_agent(self, test_queries: List[Dict] = None) -> Dict:
        """
        Evaluate agent end-to-end performance
        """
        print("🤖 Evaluating Agent Performance...")
        
        if test_queries is None:
            test_queries = self.create_test_queries()
        
        results = {
            'task_success': [],
            'relevance_scores': [],
            'price_extraction': [],
            'response_times': [],
            'sites_coverage': []
        }
        
        for test_case in tqdm(test_queries, desc="Testing queries"):
            try:
                # Execute agent
                start_time = time.time()
                agent_response = self.agent.process_request(test_case['query'])
                response_time = time.time() - start_time
                
                # Evaluate results
                search_results = agent_response.get('results', [])
                
                # Task success (at least one result)
                task_success = len(search_results) > 0
                results['task_success'].append(task_success)
                
                # Relevance evaluation
                relevance = self.evaluate_search_relevance(
                    search_results, 
                    test_case['query'],
                    test_case.get('expected_keywords', [])
                )
                results['relevance_scores'].append(relevance['avg_relevance'])
                
                # Price extraction
                price_eval = self.evaluate_price_extraction(search_results)
                results['price_extraction'].append(price_eval['price_extraction_rate'])
                
                # Response time
                results['response_times'].append(response_time)
                
                # Sites coverage
                sites_found = len(set(r.get('site', '') for r in search_results))
                results['sites_coverage'].append(sites_found)
                
                # Log individual result
                logger.info(f"Query: {test_case['query']}")
                logger.info(f"  Success: {task_success}")
                logger.info(f"  Relevance: {relevance['avg_relevance']:.2f}")
                logger.info(f"  Sites: {sites_found}")
                
            except Exception as e:
                logger.error(f"Error evaluating query '{test_case['query']}': {e}")
                results['task_success'].append(False)
                results['relevance_scores'].append(0)
                results['price_extraction'].append(0)
                results['response_times'].append(0)
                results['sites_coverage'].append(0)
        
        # Calculate summary statistics
        summary = {
            'task_success_rate': np.mean(results['task_success']),
            'avg_relevance': np.mean(results['relevance_scores']),
            'avg_price_extraction': np.mean(results['price_extraction']),
            'avg_response_time': np.mean(results['response_times']),
            'avg_sites_coverage': np.mean(results['sites_coverage']),
            'total_queries': len(test_queries)
        }
        
        return summary, results

def generate_evaluation_report(model_results: Dict, agent_results: Dict) -> str:
    """
    Generate a comprehensive evaluation report
    """
    report = """
# AI Price Comparison Agent - Evaluation Report
=========================================

## Model-Level Metrics (Image-to-Text)

### Text Generation Quality
- **BLEU Score**: {:.3f}
- **ROUGE-1**: {:.3f}
- **ROUGE-2**: {:.3f}
- **ROUGE-L**: {:.3f}

### Keyword Accuracy
- **F1 Score**: {:.3f}

### Performance
- **Avg Inference Time**: {:.3f}s
- **Total Samples Evaluated**: {}

## Agent-Level Metrics (End-to-End)

### Task Success
- **Success Rate**: {:.1%}
- **Avg Relevance Score**: {:.3f}

### Data Extraction
- **Price Extraction Rate**: {:.1%}
- **Avg Sites Coverage**: {:.1f} sites

### Performance
- **Avg Response Time**: {:.2f}s
- **Total Queries Tested**: {}

## Recommendations

""".format(
        model_results['avg_bleu'],
        model_results['avg_rouge1'],
        model_results['avg_rouge2'],
        model_results['avg_rougeL'],
        model_results['avg_keyword_f1'],
        model_results['avg_inference_time'],
        model_results['total_samples'],
        agent_results['task_success_rate'],
        agent_results['avg_relevance'],
        agent_results['avg_price_extraction'],
        agent_results['avg_sites_coverage'],
        agent_results['avg_response_time'],
        agent_results['total_queries']
    )
    
    # Add recommendations based on results
    if model_results['avg_bleu'] < 0.3:
        report += "- ⚠️ Low BLEU score indicates poor text generation. Consider fine-tuning with more data.\n"
    
    if agent_results['task_success_rate'] < 0.7:
        report += "- ⚠️ Low task success rate. Check web scraper selectors and site availability.\n"
    
    if agent_results['avg_response_time'] > 10:
        report += "- ⚠️ High response time. Consider optimizing scraping or using parallel requests.\n"
    
    if model_results['avg_bleu'] >= 0.5 and agent_results['task_success_rate'] >= 0.8:
        report += "- ✅ System performing well! Ready for production testing.\n"
    
    return report

def main():
    """
    Main evaluation function
    """
    print("🔬 AI Price Comparison Agent - Evaluation Suite")
    print("=" * 50)
    
    # Check for test data
    test_csv = Path("data/dataset/processed/test.csv")
    
    # Model evaluation
    model_summary = {'avg_bleu': 0, 'avg_rouge1': 0, 'avg_rouge2': 0, 
                    'avg_rougeL': 0, 'avg_keyword_f1': 0, 
                    'avg_inference_time': 0, 'total_samples': 0}
    
    if test_csv.exists():
        print("\n📊 Starting Model Evaluation...")
        model_evaluator = ModelEvaluator(use_finetuned=False)
        model_summary, model_details = model_evaluator.evaluate_model(
            str(test_csv), 
            num_samples=10  # Limit for quick testing
        )
    else:
        print("⚠️ Test dataset not found. Skipping model evaluation.")
    
    # Agent evaluation
    print("\n🤖 Starting Agent Evaluation...")
    agent_evaluator = AgentEvaluator(use_finetuned=False)
    agent_summary, agent_details = agent_evaluator.evaluate_agent()
    
    # Generate report
    report = generate_evaluation_report(model_summary, agent_summary)
    
    # Save report
    report_path = Path("evaluation_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print("\n" + "=" * 50)
    print(report)
    print("=" * 50)
    print(f"\n📄 Report saved to: {report_path}")
    
    # Save detailed results
    detailed_results = {
        'model_evaluation': model_summary,
        'agent_evaluation': agent_summary,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"📊 Detailed results saved to: evaluation_results.json")

if __name__ == "__main__":
    main()