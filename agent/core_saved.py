import logging
from typing import Dict, List, Union, Optional
from PIL import Image
import yaml

from models.inference import ImageToTextModel
from tools.web_scraper1 import WebScraperTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceComparisonAgent:
    def __init__(self, config_path="config/config.yaml", use_finetuned=False):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.image_model = ImageToTextModel(config_path, use_finetuned=use_finetuned)
        self.scraper = WebScraperTool(config_path)

    def process_request(self, input_data: Union[str, Image.Image]) -> Dict:
        logger.info("Starting price comparison agent...")

        search_query = self._reason(input_data)
        logger.info(f"Generated search query: '{search_query}'")

        search_plan = self._plan()
        logger.info(f"Search plan: {search_plan}")

        search_results = self._execute(search_query, search_plan)

        final_results = self._synthesize(search_results, search_query)

        return {
            'query': search_query,
            'results': final_results,
            'raw_results': search_results
        }

    def _reason(self, input_data: Union[str, Image.Image]) -> str:
        if isinstance(input_data, str):
            return input_data.strip()
        elif isinstance(input_data, Image.Image):
            return self.image_model.generate_search_query(input_data)
        else:
            raise ValueError("Input must be either text (str) or a PIL Image")

    def _plan(self) -> List[str]:
        return [site['name'] for site in self.config['scraping']['target_sites']]

    def _execute(self, query: str, sites: List[str]) -> Dict[str, List[Dict]]:
        return self.scraper.search_all_sites(query)

    def _synthesize(self, results: Dict[str, List[Dict]], query: str) -> List[Dict]:
        all_products = []
        for site, products in results.items():
            for product in products:
                if product.get('price') and product['price'] > 0:
                    product['relevance_score'] = self._calculate_relevance(product['title'], query)
                    all_products.append(product)

        all_products.sort(key=lambda x: (x.get('price', float('inf')), -x.get('relevance_score', 0)))
        return all_products

    def _calculate_relevance(self, product_title: str, query: str) -> float:
        query_words = set(query.lower().split())
        title_words = set(product_title.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(title_words))
        relevance = overlap / len(query_words)

        return relevance
