import streamlit as st
import pandas as pd
from PIL import Image
import logging
import json
import os

from agent.core import PriceComparisonAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI-Powered Price Comparison Agent",
    page_icon="🛍️",
    layout="wide"
)

@st.cache_resource
def load_agent():
    logger.info("Initializing Price Comparison Agent...")
    return PriceComparisonAgent(use_finetuned=True)

def main():
    st.title("PriceWarden: Your go-to AI Price Comparison Agent")
    st.markdown(
        "Find the best prices across multiple e-commerce platforms using AI! "
        "Upload an image or enter a product description to get started."
    )

    agent = load_agent()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Image Search")
        uploaded_file = st.file_uploader(
            "Upload a product image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of the product you want to find"
        )
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("📝 Text Search")
        text_query = st.text_input(
            "Or enter a product description",
            placeholder="e.g., Nike Air Force 1 white sneakers"
        )

    if st.button("Search for Best Prices", type="primary"):
        if not uploaded_file and not text_query:
            st.warning("Please upload an image or enter a product description.")
        else:
            with st.spinner("Searching across multiple platforms... This may take a moment."):
                try:
                    if uploaded_file:
                        input_data = Image.open(uploaded_file)
                        st.info("Analyzing image to generate search query...")
                    else:
                        input_data = text_query

                    results = agent.process_request(input_data)

                    st.success(f"**Search Query:** `{results['query']}`")

                    if results['results']:
                        st.subheader("Great!, Here are the Best Prices Found")

                        price_sorted_results = sorted(results['results'], key=lambda p: p.get('price', float('inf')))

                        df_data = [
                            {
                                'Rank': idx,
                                'Product': f"{p['title'][:60]}..." if len(p['title']) > 60 else p['title'],
                                'Price': p['price'],
                                'Site': p['site'].capitalize(),
                                'Relevance': p['relevance_score'],
                                'Link': p['link']
                            }
                            for idx, p in enumerate(price_sorted_results[:100], 1)
                        ]
                        df = pd.DataFrame(df_data)

                        st.dataframe(
                            df,
                            column_config={
                                "Price": st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
                                "Relevance": st.column_config.ProgressColumn("Relevance", format="%.2f%%", min_value=0, max_value=1),
                                "Link": st.column_config.LinkColumn("Product Link", display_text="🔗 View")
                            },
                            hide_index=True,
                            use_container_width=True,
                            height=735
                        )

                        with st.expander("Price Statistics"):
                            prices = [p['price'] for p in results['results'] if p.get('price') and p['price'] > 0]
                            if prices:
                                stat_col1, stat_col2, stat_col3 = st.columns(3)
                                stat_col1.metric("Lowest Price", f"₹{min(prices):,.2f}")
                                stat_col2.metric("Average Price", f"₹{sum(prices)/len(prices):,.2f}")
                                stat_col3.metric("Highest Price", f"₹{max(prices):,.2f}")

                        with st.expander("🔍 Detailed Results by Site"):
                            for site, products in results['raw_results'].items():
                                if products:
                                    st.write(f"**{site.capitalize()}** ({len(products)} results found)")
                                    for p in products:
                                        price_str = f"₹{p.get('price', 0.0):,.2f}" if p.get('price') else "N/A"
                                        st.markdown(f"- [{p['title'][:70]}...]({p.get('link', '#')}) - **{price_str}**")
                                else:
                                    st.write(f"**{site.capitalize()}** - No results found")

                    else:
                        st.warning("No products found for your query. Please try a different search.")

                    if results and results['results']:
                        st.info(f"**AI Agent Thoughts**: {results.get('agent_thoughts', 'No thoughts recorded.')}")

                        confidence = results.get('confidence', 0.5)
                        st.progress(confidence)
                        st.caption(f"Agent Confidence: {confidence * 100:.0f}%")

                        # Show recommendations
                        if results.get('recommendations'):
                            st.success("💡 **AI Recommendations:**")
                            for rec in results['recommendations']:
                                st.write(f"• {rec}")

                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    logger.error(f"Error during Streamlit search execution: {str(e)}", exc_info=True)


    # --- Footer ---
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
        <p>Built with ❤️ by Aekam using Streamlit, Transformers, and Web Scraping</p>
        <p style='font-size: 0.8em; color: gray;'>
        Note: This is a prototype ATM. Actual prices may vary. Always verify on the merchant's website.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.sidebar.button("Show Agent Brain(Memory Storage History)"):
        st.subheader("AI Agent Learning")

        if os.path.exists("agent_history.json"):
            with open("agent_history.json") as f:
                history = json.load(f)

            st.write("**Agent Memory:**")
            for query, data in history.items():
                st.write(f"• '{query}': searched {data['count']} times, "
                         f"success rate: {data['success_rate'] * 100:.0f}%")

if __name__ == "__main__":
    main()