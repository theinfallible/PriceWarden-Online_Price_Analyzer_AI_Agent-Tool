# PriceWarden: AI-Powered Price Comparison Agent 

An intelligent agent that uses a fine-tuned vision model to find the best prices for products from an image, automating the process of cross-platform price comparison.

---

-   **Name**: [Your Name]
-   **University**: [Your University Name]
-   **Department**: [Your Department]

### ✨ Core Features

-   **Image-to-Query**: Utilizes a fine-tuned BLIP model to generate precise search queries from product images.
-   **Multi-Platform Scraping**: Scrapes top e-commerce sites (Amazon, Flipkart, Myntra) to find product listings.
-   **Intelligent Ranking**: Sorts results by the best price and relevance to the user's query.
-   **Interactive UI**: A simple and clean user interface built with Streamlit.

### 📂 Project Structure

Here is a one-line description for each key file and directory in the repository:

| File/Directory                                | Description                                                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------- |
| `agent/`                                      | Contains the core agent logic, including the Reason-Plan-Execute cycle.           |
| `archive (2)/`                                | Stores archived files, including the initial Kaggle Fashion Dataset download.     |
| `config/`                                     | Holds configuration files, primarily `config.yaml` for model and scraper settings.|
| `data/`                                       | Used for storing datasets and model checkpoints generated during fine-tuning.     |
| `models/`                                     | Includes scripts for model inference and the fine-tuning class structure.         |
| `scraper_testing/` & `testing_dev/`           | Directories for development scripts and isolated testing of components.           |
| `tools/`                                      | Contains helper modules, most importantly the web scraping tool.                  |
| `AI Agent Architecture Document.pdf`          | A high-level design document explaining the agent's architecture.                 |
| `Comprehensive Overall Pipeline Doc.pdf`      | An end-to-end document detailing the project's data and execution pipeline.       |
| `Data Science Report – Fine-tuning ....pdf`   | A detailed report on the model fine-tuning process and evaluation metrics.        |
| `Demo Video & Implementation Screenshots`     | A file containing a video demo and key screenshots of the project in action.      |
| `agent_history.json`                          | A log file that records the agent's past actions and generated results.           |
| `app.py`                                      | The main entry point for the Streamlit web application.                           |
| `requirements.txt`                            | A list of all Python dependencies required to run the project.                    |
| `run_finetuning.py`                           | The executable script to start the model fine-tuning process on the dataset.      |
| `ai_logs`                                     | Contains logs of my conversations with AI assistants (like GPT) used for debugging and development. |

### Getting Started: Setup and Running the Agent

Follow these steps to set up and run the project on your local machine.

#### 1. Clone the Repository & Install Dependencies

First, clone the repository and install the required Python packages.

```bash
git clone [your-repository-url]
cd [repository-name]
pip install -r requirements.txt
