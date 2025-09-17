import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time
import yaml
import logging
import random
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraperTool:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.scraping_config = self.config['scraping']
        self.scraper_api_key = self.scraping_config.get('scraper_api_key')

    def search_product(self, query: str, site_name: str) -> List[Dict]:
        site_config = next(
            (site for site in self.scraping_config['target_sites']
             if site['name'] == site_name), None)
        if not site_config:
            logger.error(f"Site {site_name} not found in configuration.")
            return []
        return self._scrape_with_service(query, site_config)

    def _scrape_with_service(self, query: str, site_config: Dict) -> List[Dict]:
        if not self.scraper_api_key:
            return self._get_mock_results(query, site_config['name'])
        try:
            search_query = urllib.parse.quote_plus(query)
            target_url = f"{site_config['url']}{site_config['search_endpoint']}{search_query}"
            scraper_api_url = 'http://api.scraperapi.com'
            payload = {
                'api_key': self.scraper_api_key,
                'url': target_url,
                'country_code': 'in'
            }
            response = requests.get(
                scraper_api_url,
                params=payload,
                timeout=self.scraping_config.get('timeout', 50)
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            products = self._parse_products(soup, site_config['name'])
            return products[:self.scraping_config['max_results_per_site']]
        except requests.exceptions.RequestException:
            return self._get_mock_results(query, site_config['name'])

    def _parse_products(self, soup: BeautifulSoup, site_name: str) -> List[Dict]:
        if site_name == 'amazon':
            return self._parse_amazon(soup)
        elif site_name == 'flipkart':
            return self._parse_flipkart(soup)
        elif site_name == 'myntra':
            return self._parse_myntra(soup)
        else:
            return []

    def _parse_amazon(self, soup: BeautifulSoup) -> List[Dict]:
        products = []
        containers = soup.select('div[data-component-type="s-search-result"]')
        if not containers:
            return []
        for container in containers:
            try:
                title_elem = (
                        container.select_one('h2.a-text-normal span') or
                        container.select_one('h2 a span') or
                        container.select_one('h2 span')
                )
                price_elem = (
                        container.select_one('span.a-price-whole') or
                        container.select_one('span.a-price > span.a-offscreen') or
                        container.select_one('span.a-price-range')
                )
                link_elem = None
                h2_elem = container.select_one('h2.a-text-normal') or container.select_one('h2')
                if h2_elem:
                    link_elem = h2_elem.find_parent('a') or h2_elem.select_one('a')
                if not link_elem:
                    link_elem = container.select_one('a.a-link-normal.s-no-outline')
                if not all([title_elem, price_elem, link_elem]):
                    continue
                product_link = link_elem.get('href', '')
                if product_link and not product_link.startswith('http'):
                    product_link = urllib.parse.urljoin('https://www.amazon.in', product_link)
                product = {
                    'site': 'amazon',
                    'title': title_elem.text.strip(),
                    'price': self._extract_price(price_elem),
                    'link': product_link
                }
                if product['price'] > 0:
                    products.append(product)
            except:
                continue
        return products

    def _parse_flipkart(self, soup: BeautifulSoup) -> List[Dict]:
        products = []
        possible_containers = []
        container_selectors = [
            'div[data-id]', 'div._1AtVbE', 'div._2kHMtA', 'div._4ddWXP', 'div._1xHGtK',
            'a._1fQZEK', 'div._13oc-S', 'div._1YokD2', 'div[style*="flex"]'
        ]
        seen_texts = set()
        for selector in container_selectors:
            containers = soup.select(selector)
            for container in containers:
                container_text = container.get_text()
                if '₹' in container_text or 'Rs' in container_text or any(char.isdigit() for char in container_text):
                    text_snippet = container_text[:100]
                    if text_snippet not in seen_texts:
                        possible_containers.append(container)
                        seen_texts.add(text_snippet)
        for container in possible_containers:
            try:
                title, brand, price, link = "", "", 0.0, ""
                for sel in ['a.s1Q9rs','div._4rR01T','a.WKTcLC','div.KzDlHZ','a[title]','div._2B099V a','div[class*="title"]','a.IRpwTa']:
                    elem = container.select_one(sel)
                    if elem:
                        title = elem.get('title', '') or elem.get_text(strip=True)
                        if title: break
                for sel in ['div.syl9yP','div._2WkVRV','div._2B099V','span.G6XhRU','div[class*="brand"]']:
                    elem = container.select_one(sel)
                    if elem:
                        brand = elem.get_text(strip=True)
                        if brand: break
                for sel in ['div._30jeq3','div.Nx9bqj','div._1vC4OE','div._25b18c div:first-child','span._2-ut7f','div[class*="price"]']:
                    elem = container.select_one(sel)
                    if elem:
                        price = self._extract_price_from_text(elem.get_text(strip=True))
                        if price > 0: break
                for sel in ['a._1fQZEK','a.s1Q9rs','a.WKTcLC','a[href*="/p/"]','a[href*="pid="]','a.IRpwTa','a.rPDeLR']:
                    elem = container.select_one(sel)
                    if elem and elem.get('href'):
                        link = elem.get('href')
                        break
                if not title:
                    all_text = container.get_text(separator=' ', strip=True)
                    import re
                    all_text = re.sub(r'₹[\s\d,]+', '', all_text)
                    all_text = re.sub(r'\d+%\s*OFF', '', all_text)
                    if len(all_text) > 10: title = all_text[:100]
                full_title = f"{brand} {title}" if brand else title
                full_title = ' '.join(full_title.split()).strip()
                if link and not link.startswith('http'):
                    link = f"https://www.flipkart.com{link}"
                if full_title and price > 0:
                    product = {'site':'flipkart','title':full_title[:150],'price':price,'link':link or f"https://www.flipkart.com/search?q={urllib.parse.quote(full_title)}"}
                    if not any(p['title']==product['title'] and p['price']==product['price'] for p in products):
                        products.append(product)
            except:
                continue
        products.sort(key=lambda x: (bool(x['link']), -x['price']))
        return products[:self.scraping_config.get('max_results_per_site', 10)]

    def _parse_myntra(self, soup: BeautifulSoup) -> List[Dict]:
        products = []
        containers = soup.select('a[data-refreshpage="true"]') or soup.select('li.product-base') or soup.select('div.product-base')
        for container in containers[:self.scraping_config.get('max_results_per_site', 10)]:
            try:
                brand_text, product_text, price_value, product_url = "", "", 0.0, ""
                href = container.get('href', '') if container.name=='a' else (container.select_one('a[href]') or {}).get('href','')
                if href and not href.startswith('http'): product_url=f"https://www.myntra.com/{href.lstrip('/')}"
                product_meta = container.select_one('div.product-productMetaInfo') or container
                for sel in ['h3.product-brand','div.product-brand','.product-brand','h3','span[class*="brand"]']:
                    elem = product_meta.select_one(sel)
                    if elem: brand_text=elem.get_text(strip=True); break
                for sel in ['h4.product-product','div.product-product','.product-product','h4','div[class*="product-title"]','span[class*="product-name"]']:
                    elem = product_meta.select_one(sel)
                    if elem and elem.get_text(strip=True)!=brand_text: product_text=elem.get_text(strip=True); break
                for sel in ['span.product-discountedPrice','div.product-price span','.product-discountedPrice','span[class*="price"]','div[class*="price"] span']:
                    elem = product_meta.select_one(sel)
                    if elem: price_value=self._extract_myntra_price(elem.get_text(strip=True)); break
                if price_value<=0:
                    import re
                    all_text=product_meta.get_text()
                    matches = re.findall(r'Rs\.?\s*(\d+)', all_text) or re.findall(r'₹\s*(\d+)', all_text) or re.findall(r'\b(\d{3,4})\b', all_text)
                    for m in matches: p=float(m);
                    if 100<=p<=50000: price_value=p; break
                if not (brand_text or product_text):
                    img = container.select_one('img')
                    if img: t=img.get('alt','') or img.get('title',''); words=t.split(); brand_text=words[0] if len(words)>=2 else ""; product_text=' '.join(words[1:]) if len(words)>=2 else t
                full_title=f"{brand_text} {product_text}" if brand_text and product_text else product_text or brand_text
                full_title=' '.join(full_title.split()).strip()
                if full_title and price_value>0:
                    products.append({'site':'myntra','title':full_title[:100],'price':price_value,'link':product_url or f"https://www.myntra.com/search?q={urllib.parse.quote(full_title)}"})
            except:
                continue
        if not products: return self._myntra_fallback_parsing(soup)
        return products

    def _myntra_fallback_parsing(self, soup: BeautifulSoup) -> List[Dict]:
        products=[]
        price_patterns = soup.find_all(text=lambda t:t and ('Rs.' in str(t) or '₹' in str(t)))
        for price_text in price_patterns[:20]:
            try:
                price_value=self._extract_myntra_price(str(price_text))
                if price_value<=0 or price_value>50000: continue
                container=price_text.parent
                for _ in range(5):
                    if container and container.name in ['a','li','div','article'] and 'product' in container.get('class',[]): break
                    container=container.parent if container else None
                if not container: continue
                img=container.find('img')
                title=img.get('alt','') or img.get('title','') if img else ""
                if not title:
                    for t in container.find_all(text=True):
                        t=t.strip()
                        if len(t)>10 and not any(p in t for p in ['Rs','₹','%','OFF']) and not t.isdigit():
                            title=t; break
                if not title: title=f"Myntra Product @ ₹{price_value}"
                link=container.get('href','') if container.name=='a' else (container.find('a') or {}).get('href','')
                if link and not link.startswith('http'): link=f"https://www.myntra.com/{link.lstrip('/')}"
                products.append({'site':'myntra','title':title[:100],'price':price_value,'link':link or f"https://www.myntra.com/search?q={title.replace(' ','+')}"})
                if len(products)>=10: break
            except: continue
        return products

    def _extract_myntra_price(self, price_text:str)->float:
        if not price_text: return 0.0
        import re
        price_text=str(price_text).strip()
        patterns=[r'Rs\.\s*(\d+)',r'₹\s*(\d+)',r'Rs\s+(\d+)',r'(\d+\.\d+)',r'(\d{1,3}(?:,\d{3})*)',r'(\d+)']
        for pat in patterns:
            m=re.search(pat,price_text)
            if m: return float(m.group(1).replace(',',''))
        digits=''.join(filter(str.isdigit,price_text))
        return float(digits) if digits else 0.0

    def _extract_price_from_text(self,text:str)->float:
        if not text: return 0.0
        import re
        text=text.strip()
        patterns=[r'₹\s*([0-9,]+)', r'Rs\.?\s*([0-9,]+)', r'([0-9]{1,2},?[0-9]{3,})', r'([0-9]+)']
        for pat in patterns:
            m=re.search(pat,text)
            if m: p=float(m.group(1).replace(',',''));
            if 10<=p<=1000000: return p
        digits=''.join(filter(str.isdigit,text))
        return float(digits) if digits and 10<=float(digits)<=1000000 else 0.0

    def _extract_price(self, price_element)->float:
        if not price_element: return 0.0
        try:
            price_text=price_element.text.strip()
            cleaned=''.join(filter(lambda c:c.isdigit() or c=='.',price_text.replace(',','')))
            if '-' in price_text: cleaned=cleaned.split('-')[0]
            return float(cleaned) if cleaned else 0.0
        except: return 0.0

    def search_all_sites(self, query:str)->Dict[str,List[Dict]]:
        results={}
        for site in self.scraping_config['target_sites']:
            site_name=site['name']
            results[site_name]=self.search_product(query, site_name)
            time.sleep(random.uniform(0.5,1.5))
        return results

    def _get_mock_results(self, query:str, site_name:str)->List[Dict]:
        mock_data={
            'amazon':[{'site':'amazon','title':f'{query} - Premium Edition','price':7495.00,'link':'https://www.amazon.in/mock-product'},
                     {'site':'amazon','title':f'{query} - Standard Version','price':5995.00,'link':'https://www.amazon.in/mock-product-2'}],
            'flipkart':[{'site':'flipkart','title':f'{query} - Best Seller','price':6999.00,'link':'https://www.flipkart.com/mock-product'},
                        {'site':'flipkart','title':f'{query} - New Arrival','price':8499.00,'link':'https://www.flipkart.com/mock-product-2'}],
            'myntra':[{'site':'myntra','title':f'{query} - Trending','price':7195.00,'link':'https://www.myntra.com/mock-product'},
                      {'site':'myntra','title':f'{query} - Limited Edition','price':9995.00,'link':'https://www.myntra.com/mock-product-2'}]
        }
        results=mock_data.get(site_name,[])
        for item in results:
            item['price']+=random.randint(-500,500)
            item['price']=max(item['price'],1000)
        return results
