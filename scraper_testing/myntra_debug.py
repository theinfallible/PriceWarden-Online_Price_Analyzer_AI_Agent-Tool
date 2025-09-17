#!/usr/bin/env python3

from bs4 import BeautifulSoup
import re


def debug_myntra_html():
    try:
        with open('myntra_response.html', 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        print(f"Total HTML length: {len(html_content)}")

        containers = soup.select('a[data-refreshpage="true"]')
        print(f"\nFound {len(containers)} product containers")

        if not containers:
            print("No containers found! Let's check what selectors might work...")
            all_links = soup.find_all('a', href=True)
            print(f"Total links found: {len(all_links)}")

            product_links = [link for link in all_links if '/buy' in link.get('href', '')]
            print(f"Links with '/buy': {len(product_links)}")
            return

        for i, container in enumerate(containers[:3]):
            print(f"\n{'=' * 50}")
            print(f"CONTAINER {i + 1}:")
            print(f"{'=' * 50}")

            container_html = str(container)
            print(f"Container HTML length: {len(container_html)}")
            print(f"First 500 chars:\n{container_html[:500]}...")

            print(f"\n--- ELEMENT ANALYSIS ---")

            brand_selectors = ['h3.product-brand', '.product-brand', 'h3']
            for selector in brand_selectors:
                elem = container.select_one(selector)
                if elem:
                    print(f"Brand ({selector}): '{elem.get_text().strip()}'")
                    break
            else:
                print("Brand: Not found with any selector")

            product_selectors = ['h4.product-product', '.product-product', 'h4']
            for selector in product_selectors:
                elem = container.select_one(selector)
                if elem:
                    print(f"Product ({selector}): '{elem.get_text().strip()}'")
                    break
            else:
                print("Product name: Not found with any selector")

            price_selectors = ['span.product-discountedPrice', '.product-discountedPrice', 'span']
            for selector in price_selectors:
                elems = container.select(selector)
                if elems:
                    for elem in elems:
                        text = elem.get_text().strip()
                        if 'Rs' in text or '₹' in text:
                            print(f"Price ({selector}): '{text}'")
                            break
                    else:
                        continue
                    break
            else:
                print("Price: Not found with any selector")

            href = container.get('href', '')
            print(f"Href: '{href}'")

            img = container.select_one('img')
            if img:
                alt_text = img.get('alt', '')
                src = img.get('src', '')
                print(f"Image alt: '{alt_text[:100]}...' (truncated)")
                print(f"Image src: '{src[:100]}...' (truncated)")
            else:
                print("Image: Not found")

            all_text = container.get_text(separator=' ', strip=True)
            print(f"All text: '{all_text[:200]}...' (truncated)")

            price_patterns = [r'Rs\.?\s*(\d+)', r'₹\s*(\d+)', r'(\d+)']
            for pattern in price_patterns:
                matches = re.findall(pattern, all_text)
                if matches:
                    print(f"Price pattern '{pattern}' found: {matches}")
                    break

        print(f"\n{'=' * 50}")
        print("COMMON CLASSES IN CONTAINERS:")
        print(f"{'=' * 50}")

        all_classes = set()
        for container in containers:
            for elem in container.find_all(True):
                if elem.get('class'):
                    all_classes.update(elem.get('class'))

        product_related_classes = [cls for cls in all_classes if 'product' in cls.lower()]
        price_related_classes = [cls for cls in all_classes if 'price' in cls.lower()]

        print(f"Product-related classes: {sorted(product_related_classes)}")
        print(f"Price-related classes: {sorted(price_related_classes)}")

        other_classes = [cls for cls in all_classes if 'product' not in cls.lower() and 'price' not in cls.lower()]
        print(f"Other classes (sample): {sorted(list(other_classes))[:20]}")

    except FileNotFoundError:
        print("ERROR: myntra_response.html file not found!")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    debug_myntra_html()
