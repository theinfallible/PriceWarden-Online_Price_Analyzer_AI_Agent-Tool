import logging
from typing import Dict, List, Union, Optional
from PIL import Image
import yaml
import time
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class PriceComparisonAgent:
    """Enhanced agent with more autonomous decision-making capabilities"""

    def __init__(self, config_path="config/config.yaml", use_finetuned=True):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        from models.inference import ImageToTextModel
        from tools.web_scraper1 import WebScraperTool

        self.image_model = ImageToTextModel(config_path, use_finetuned=use_finetuned)
        self.scraper = WebScraperTool(config_path)

        self.search_history = self._load_history()
        self.user_preferences = self._load_preferences()

        self.confidence_threshold = self.config['agent']['confidence_threshold']
        self.max_retries = self.config['agent']['max_retries']

    def process_request(self, input_data: Union[str, Image.Image],
                        user_context: Dict = None) -> Dict:
        """
        Enhanced agentic processing with context awareness and learning
        """
        logger.info("AI Agent starting intelligent price comparison...")

        search_query, intent = self._understand_request(input_data, user_context)
        logger.info(f"Agent understood: Query='{search_query}', Intent='{intent}'")

        search_strategy = self._create_strategy(search_query, intent)
        logger.info(f"Agent strategy: {search_strategy}")

        search_results = self._intelligent_execute(search_query, search_strategy)

        analysis = self._analyze_results(search_results, search_query, intent)

        recommendations = self._generate_recommendations(analysis, user_context)

        self._learn_from_search(search_query, analysis)

        return {
            'query': search_query,
            'intent': intent,
            'results': analysis['best_results'],
            'raw_results': analysis.get('raw_results', {}),
            'insights': analysis['insights'],
            'recommendations': recommendations,
            'confidence': analysis['confidence'],
            'agent_thoughts': self._get_agent_thoughts(search_query, analysis)
        }

    def _understand_request(self, input_data: Union[str, Image.Image],
                            user_context: Dict = None) -> tuple:
        """
        Advanced understanding of user request with intent detection
        """
        if isinstance(input_data, str):
            search_query = input_data.strip()
        else:
            search_query = self.image_model.generate_search_query(input_data)

        intent = self._detect_intent(search_query)

        if search_query in self.search_history:
            past_data = self.search_history[search_query]
            if past_data['success_rate'] < 0.5:
                search_query = self._refine_query(search_query)
                logger.info(f"Agent refined query based on past experience: {search_query}")

        return search_query, intent

    def _detect_intent(self, query: str) -> str:
        """
        Detect user intent from query
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ['cheap', 'budget', 'affordable', 'lowest']):
            return 'budget_conscious'
        elif any(word in query_lower for word in ['premium', 'luxury', 'best', 'quality']):
            return 'quality_focused'
        elif any(word in query_lower for word in ['gift', 'present']):
            return 'gift_shopping'
        elif any(word in query_lower for word in ['urgent', 'fast', 'quick', 'today']):
            return 'urgent_need'
        else:
            return 'general_search'

    def _create_strategy(self, query: str, intent: str) -> Dict:
        """
        Create intelligent search strategy based on intent
        """
        strategy = {
            'sites': [],
            'priority': '',
            'filters': {},
            'max_price': None,
            'min_price': None
        }

        if intent == 'budget_conscious':
            strategy['sites'] = ['flipkart', 'amazon', 'myntra']
            strategy['priority'] = 'lowest_price'
            strategy['max_price'] = 5000
        elif intent == 'quality_focused':
            strategy['sites'] = ['amazon', 'myntra', 'flipkart']
            strategy['priority'] = 'best_rated'
            strategy['min_price'] = 1000
        elif intent == 'gift_shopping':
            strategy['sites'] = ['myntra', 'amazon', 'flipkart']
            strategy['priority'] = 'popular'
        else:
            strategy['sites'] = ['amazon', 'flipkart', 'myntra']
            strategy['priority'] = 'balanced'

        return strategy

    def _intelligent_execute(self, query: str, strategy: Dict) -> Dict[str, List[Dict]]:
        """
        Execute searches with retry logic and error handling
        """
        results = {}

        for site in strategy['sites']:
            attempts = 0
            while attempts < self.max_retries:
                try:
                    logger.info(f"Agent searching {site} (attempt {attempts + 1})")
                    site_results = self.scraper.search_product(query, site)

                    if site_results:
                        filtered = self._apply_filters(site_results, strategy)
                        results[site] = filtered
                        break
                    else:
                        attempts += 1
                        if attempts < self.max_retries:
                            time.sleep(2)

                except Exception as e:
                    logger.warning(f"Agent encountered error on {site}: {e}")
                    attempts += 1

            if site not in results:
                results[site] = []

        return results

    def _apply_filters(self, products: List[Dict], strategy: Dict) -> List[Dict]:
        """
        Apply intelligent filters based on strategy
        """
        filtered = products

        if strategy.get('max_price'):
            filtered = [p for p in filtered if p['price'] <= strategy['max_price']]

        if strategy.get('min_price'):
            filtered = [p for p in filtered if p['price'] >= strategy['min_price']]

        if strategy['priority'] == 'lowest_price':
            filtered.sort(key=lambda x: x['price'])
        elif strategy['priority'] == 'best_rated':
            filtered.sort(key=lambda x: x['price'], reverse=True)

        return filtered

    def _analyze_results(self, results: Dict[str, List[Dict]],
                         query: str, intent: str) -> Dict:
        """
        Deep analysis of search results with insights
        """
        all_products = []
        for site, products in results.items():
            for product in products:
                product['relevance_score'] = self._calculate_relevance(product['title'], query)

            all_products.extend(products)

        if not all_products:
            return {
                'best_results': [],
                'insights': {'error': 'No products found'},
                'confidence': 0.0,
                'raw_results': results
            }

        prices = [p['price'] for p in all_products if p['price'] > 0]

        insights = {
            'total_products': len(all_products),
            'price_range': {
                'min': min(prices) if prices else 0,
                'max': max(prices) if prices else 0,
                'avg': sum(prices) / len(prices) if prices else 0
            },
            'best_deal_site': self._find_best_deal_site(results),
            'price_trend': self._analyze_price_trend(query, prices),
            'recommendation_confidence': self._calculate_confidence(all_products, query)
        }

        best_results = self._select_best_results(all_products, intent)

        return {
            'best_results': best_results,
            'insights': insights,
            'confidence': insights['recommendation_confidence'],
            'raw_results': results
        }

    def _find_best_deal_site(self, results: Dict[str, List[Dict]]) -> str:
        """
        Find which site has the best deals
        """
        site_avg_prices = {}

        for site, products in results.items():
            if products:
                prices = [p['price'] for p in products if p['price'] > 0]
                if prices:
                    site_avg_prices[site] = sum(prices) / len(prices)

        if site_avg_prices:
            return min(site_avg_prices, key=site_avg_prices.get)
        return "unknown"

    def _analyze_price_trend(self, query: str, current_prices: List[float]) -> str:
        """
        Analyze price trends based on historical data
        """
        if query in self.search_history:
            past_avg = self.search_history[query].get('avg_price', 0)
            current_avg = sum(current_prices) / len(current_prices) if current_prices else 0

            if current_avg < past_avg * 0.9:
                return "prices_dropping"
            elif current_avg > past_avg * 1.1:
                return "prices_rising"
            else:
                return "prices_stable"
        return "no_history"

    def _calculate_confidence(self, products: List[Dict], query: str) -> float:
        """
        Calculate confidence score for recommendations
        """
        confidence = 0.0

        confidence += min(len(products) / 20, 0.3)

        sites = set(p['site'] for p in products)
        confidence += min(len(sites) / 3, 0.3)

        if products:
            prices = [p['price'] for p in products if p['price'] > 0]
            if prices and max(prices) / min(prices) > 1.2:
                confidence += 0.2

        query_words = set(query.lower().split())
        matching_products = sum(
            1 for p in products
            if any(word in p['title'].lower() for word in query_words)
        )
        confidence += min(matching_products / len(products), 0.2) if products else 0

        return min(confidence, 1.0)

    def _select_best_results(self, products: List[Dict], intent: str) -> List[Dict]:
        """
        Select best products based on intent
        """
        if intent == 'budget_conscious':
            products.sort(key=lambda x: x['price'])
            return products[:5]
        elif intent == 'quality_focused':
            products.sort(key=lambda x: x['price'], reverse=True)
            return products[:5]
        else:
            products.sort(key=lambda x: x['price'])
            cheap = products[:2]
            mid = products[len(products) // 2:len(products) // 2 + 2]
            expensive = products[-1:]
            return cheap + mid + expensive

    def _generate_recommendations(self, analysis: Dict,
                                  user_context: Dict = None) -> List[str]:
        """
        Generate AI-powered recommendations
        """
        recommendations = []
        insights = analysis['insights']

        if insights.get('price_range'):
            price_range = insights['price_range']
            if price_range['max'] / price_range['min'] > 3:
                recommendations.append(
                    f"Large price variation detected (₹{price_range['min']:.0f} - ₹{price_range['max']:.0f}). "
                    f"The mid-range options around ₹{price_range['avg']:.0f} often offer best value."
                )

        if insights.get('best_deal_site'):
            recommendations.append(
                f"{insights['best_deal_site'].title()} has the best average prices for this search."
            )

        if insights.get('price_trend') == 'prices_dropping':
            recommendations.append(
                "Prices are trending down compared to historical data. Good time to buy!"
            )
        elif insights.get('price_trend') == 'prices_rising':
            recommendations.append(
                "Prices are higher than usual. Consider waiting or looking for deals."
            )

        if analysis['confidence'] < 0.5:
            recommendations.append(
                "Limited results found. Try a more general search term for better options."
            )

        return recommendations

    def _learn_from_search(self, query: str, analysis: Dict):
        """
        Learn from this search for future improvements
        """
        if query not in self.search_history:
            self.search_history[query] = {
                'count': 0,
                'avg_price': 0,
                'success_rate': 0
            }

        self.search_history[query]['count'] += 1

        if analysis['insights'].get('price_range'):
            self.search_history[query]['avg_price'] = analysis['insights']['price_range']['avg']

        self.search_history[query]['success_rate'] = analysis['confidence']
        self.search_history[query]['last_searched'] = datetime.now().isoformat()

        self._save_history()

    def _get_agent_thoughts(self, query: str, analysis: Dict) -> str:
        """
        Generate human-readable agent thoughts about the search
        """
        thoughts = []

        if analysis['confidence'] > 0.7:
            thoughts.append(f"I'm confident about these results for '{query}'")
        else:
            thoughts.append(f"Results for '{query}' are limited, but I've found the best available options")

        if analysis['insights'].get('best_deal_site'):
            thoughts.append(f"I noticed {analysis['insights']['best_deal_site']} has competitive prices")

        if analysis['insights'].get('price_trend') == 'prices_dropping':
            thoughts.append("Prices seem lower than usual - great timing!")

        return ". ".join(thoughts)

    def _refine_query(self, query: str) -> str:
        """
        Refine query based on past failures
        """
        if len(query.split()) > 3:
            words = query.split()[:2]
            return " ".join(words)
        else:
            if "kurta" in query:
                return "women kurta ethnic wear"
            return query + " online shopping"

    def _load_history(self) -> Dict:
        """Load search history from file"""
        history_file = "agent_history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_history(self):
        """Save search history to file"""
        with open("agent_history.json", 'w') as f:
            json.dump(self.search_history, f)

    def _load_preferences(self) -> Dict:
        """Load user preferences"""
        return {
            'preferred_sites': ['amazon', 'flipkart', 'myntra'],
            'budget_range': (500, 10000),
            'preferred_brands': []
        }

    def _calculate_relevance(self, product_title: str, query: str) -> float:
        """
        Calculates a simple relevance score based on word overlap
        between the product title and the search query.
        """
        query_words = set(query.lower().split())
        title_words = set(product_title.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(title_words))
        relevance = overlap / len(query_words)

        return relevance
