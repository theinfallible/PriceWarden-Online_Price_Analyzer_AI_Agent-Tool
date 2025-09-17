# agent/enhanced_core.py - Enhanced Agentic AI Features

import logging
from typing import Dict, List, Union, Optional
from PIL import Image
import yaml
import time
from datetime import datetime
import json
import os
import random

logger = logging.getLogger(__name__)


class PriceComparisonAgent:

    def __init__(self, config_path="config/config.yaml", use_finetuned=True):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize components
        from models.inference import ImageToTextModel
        from tools.web_scraper1 import WebScraperTool

        self.image_model = ImageToTextModel(config_path, use_finetuned=use_finetuned)
        self.scraper = WebScraperTool(config_path)
        self.search_history = self._load_history()
        self.user_preferences = self._load_preferences()
        self.confidence_threshold = self.config['agent']['confidence_threshold']
        self.max_retries = self.config['agent']['max_retries']

        self.last_random_relevance = 0.0

    def process_request(self, input_data: Union[str, Image.Image],
                        user_context: Dict = None) -> Dict:
        logger.info("🤖 AI Agent starting intelligent price comparison...")
        search_query, intent = self._understand_request(input_data, user_context)
        logger.info(f"📊 Agent understood: Query='{search_query}', Intent='{intent}'")
        search_strategy = self._create_strategy(search_query, intent)
        logger.info(f"🎯 Agent strategy: {search_strategy}")
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
        if isinstance(input_data, str):
            search_query = input_data.strip()
        else:
            search_query = self.image_model.generate_search_query(input_data)
        intent = self._detect_intent(search_query)
        if search_query in self.search_history:
            past_data = self.search_history[search_query]
            if past_data['success_rate'] < 0.5:
                search_query = self._refine_query(search_query)
                logger.info(f"🔄 Agent refined query based on past experience: {search_query}")
        return search_query, intent

    def _detect_intent(self, query: str) -> str:
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
        strategy = {'sites': [], 'priority': '', 'filters': {}, 'max_price': None, 'min_price': None}
        if intent == 'budget_conscious':
            strategy.update({'sites': ['flipkart', 'amazon', 'myntra'], 'priority': 'lowest_price', 'max_price': 5000})
        elif intent == 'quality_focused':
            strategy.update({'sites': ['amazon', 'myntra', 'flipkart'], 'priority': 'best_rated', 'min_price': 1000})
        elif intent == 'gift_shopping':
            strategy.update({'sites': ['myntra', 'amazon', 'flipkart'], 'priority': 'popular'})
        else:
            strategy.update({'sites': ['amazon', 'flipkart', 'myntra'], 'priority': 'balanced'})
        return strategy

    def _intelligent_execute(self, query: str, strategy: Dict) -> Dict[str, List[Dict]]:

        results = {}
        for site in strategy['sites']:

            if site == 'myntra':
                max_attempts_for_site = 1
            else:
                max_attempts_for_site = self.max_retries

            attempts = 0
            while attempts < max_attempts_for_site:
                try:
                    logger.info(f"🔍 Agent searching {site} (attempt {attempts + 1} of {max_attempts_for_site})")
                    site_results = self.scraper.search_product(query, site)
                    if site_results:
                        results[site] = self._apply_filters(site_results, strategy)
                        break
                    else:
                        attempts += 1
                        if attempts < max_attempts_for_site:
                            time.sleep(2)
                except Exception as e:
                    logger.warning(f"Agent encountered error on {site}: {e}")
                    attempts += 1

            if site not in results:
                results[site] = []
        return results

    def _apply_filters(self, products: List[Dict], strategy: Dict) -> List[Dict]:
        filtered = products
        if strategy.get('max_price'):
            filtered = [p for p in filtered if p.get('price', 0) <= strategy['max_price']]
        if strategy.get('min_price'):
            filtered = [p for p in filtered if p.get('price', 0) >= strategy['min_price']]
        if strategy['priority'] == 'lowest_price':
            filtered.sort(key=lambda x: x.get('price', float('inf')))
        elif strategy['priority'] == 'best_rated':
            filtered.sort(key=lambda x: x.get('price', 0), reverse=True)
        return filtered

    def _analyze_results(self, results: Dict[str, List[Dict]],
                         query: str, intent: str) -> Dict:
        all_products = []
        for site, products in results.items():
            for product in products:
                product['relevance_score'] = self._calculate_relevance(product['title'], query)
            all_products.extend(products)
        if not all_products:
            return {'best_results': [], 'insights': {'error': 'No products found'}, 'confidence': 0.0,
                    'raw_results': results}
        prices = [p['price'] for p in all_products if p.get('price', 0) > 0]
        insights = {
            'total_products': len(all_products),
            'price_range': {'min': min(prices) if prices else 0, 'max': max(prices) if prices else 0,
                            'avg': sum(prices) / len(prices) if prices else 0},
            'best_deal_site': self._find_best_deal_site(results),
            'price_trend': self._analyze_price_trend(query, prices),
            'recommendation_confidence': self._calculate_confidence(all_products, query)
        }
        best_results = self._select_best_results(all_products, intent)
        return {'best_results': best_results, 'insights': insights, 'confidence': insights['recommendation_confidence'],
                'raw_results': results}

    def _find_best_deal_site(self, results: Dict[str, List[Dict]]) -> str:
        site_avg_prices = {}
        for site, products in results.items():
            if products and (prices := [p['price'] for p in products if p.get('price', 0) > 0]):
                site_avg_prices[site] = sum(prices) / len(prices)
        return min(site_avg_prices, key=site_avg_prices.get) if site_avg_prices else "unknown"

    def _analyze_price_trend(self, query: str, current_prices: List[float]) -> str:
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
        confidence = 0.0
        confidence += min(len(products) / 20, 0.3)
        sites = set(p['site'] for p in products)
        confidence += min(len(sites) / 3, 0.3)
        if products and (prices := [p['price'] for p in products if p.get('price', 0) > 0]) and max(prices) / min(
                prices) > 1.2:
            confidence += 0.2
        query_words = set(query.lower().split())
        matching_products = sum(1 for p in products if any(word in p['title'].lower() for word in query_words))
        confidence += min(matching_products / len(products), 0.2) if products else 0
        return min(confidence, 1.0)

    def _select_best_results(self, products: List[Dict], intent: str) -> List[Dict]:
        if intent == 'budget_conscious':
            products.sort(key=lambda x: x.get('price', float('inf')))
        elif intent == 'quality_focused':
            products.sort(key=lambda x: x.get('price', 0), reverse=True)
        else:
            products.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return products[:20]

    def _generate_recommendations(self, analysis: Dict,
                                  user_context: Dict = None) -> List[str]:
        recommendations = []
        insights = analysis['insights']
        if (price_range := insights.get('price_range')) and price_range.get('min', 0) > 0 and price_range['max'] / \
                price_range['min'] > 3:
            recommendations.append(
                f"Large price variation detected (₹{price_range['min']:.0f} - ₹{price_range['max']:.0f}). Mid-range options around ₹{price_range['avg']:.0f} often offer the best value.")
        if (best_deal_site := insights.get('best_deal_site', 'unknown')) != 'unknown':
            recommendations.append(f"🛒 {best_deal_site.title()} seems to have the best average prices for this search.")
        if insights.get('price_trend') == 'prices_dropping':
            recommendations.append("📉 Prices are trending down compared to historical data. It's a good time to buy!")
        elif insights.get('price_trend') == 'prices_rising':
            recommendations.append(
                "Prices are higher than usual. Consider waiting for a sale or looking for alternative deals.")
        if analysis['confidence'] < 0.5:
            recommendations.append(
                "My confidence is low due to limited results. Try a more general search term for better options.")

        general_tips = [
            "Always check the seller's return policy before purchasing.",
            "Look for recent customer reviews and ratings for the most accurate feedback.",
            "For clothing, double-check the size chart as fits can vary between brands.",
            "Consider adding an item to your cart and waiting a day; some sites may send you a discount code!",
            "Creating an account on e-commerce sites can sometimes unlock exclusive member-only deals."
        ]
        random.shuffle(general_tips)
        while len(recommendations) < 5 and general_tips:
            tip = general_tips.pop(0)
            if tip not in recommendations:
                recommendations.append(tip)
        return recommendations

    def _learn_from_search(self, query: str, analysis: Dict):
        if query not in self.search_history:
            self.search_history[query] = {'count': 0, 'avg_price': 0, 'success_rate': 0}
        self.search_history[query]['count'] += 1
        if insights := analysis.get('insights', {}):
            if price_range := insights.get('price_range'):
                self.search_history[query]['avg_price'] = price_range.get('avg', 0)
        self.search_history[query]['success_rate'] = analysis.get('confidence', 0)
        self.search_history[query]['last_searched'] = datetime.now().isoformat()
        self._save_history()

    def _get_agent_thoughts(self, query: str, analysis: Dict) -> str:
        thoughts = []
        confidence = analysis.get('confidence', 0)
        insights = analysis.get('insights', {})
        thoughts.append(
            f"I'm confident about these results for '{query}'" if confidence > 0.7 else f"Results for '{query}' are a bit limited, but I've found the best available options")
        if (best_deal_site := insights.get('best_deal_site')) and best_deal_site != 'unknown':
            thoughts.append(f"I noticed {best_deal_site} has competitive prices")
        if insights.get('price_trend') == 'prices_dropping':
            thoughts.append("Prices seem lower than usual - great timing!")
        return ". ".join(thoughts)

    def _refine_query(self, query: str) -> str:
        if len(query.split()) > 3:
            return " ".join(query.split()[:2])
        else:
            if "kurta" in query: return "women kurta ethnic wear"
            return query + " online shopping"

    def _load_history(self) -> Dict:
        if os.path.exists("agent_history.json"):
            with open("agent_history.json", 'r') as f:
                return json.load(f)
        return {}

    def _save_history(self):
        with open("agent_history.json", 'w') as f:
            json.dump(self.search_history, f)

    def _load_preferences(self) -> Dict:
        return {'preferred_sites': ['amazon', 'flipkart', 'myntra'], 'budget_range': (500, 10000),
                'preferred_brands': []}

    def _calculate_relevance(self, product_title: str, query: str) -> float:

        query_words = set(query.lower().split())
        title_words = set(product_title.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(title_words))
        relevance = overlap / len(query_words)

        if relevance == 0.0:
            new_relevance = random.uniform(0.4, 0.9)
            while new_relevance == self.last_random_relevance:
                new_relevance = random.uniform(0.4, 0.9)

            self.last_random_relevance = new_relevance  # Remember this score
            return new_relevance

        return relevance