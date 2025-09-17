# PriceWarden: AI-Powered Price Comparison Agent 

An intelligent agent that uses a fine-tuned vision model to find the best prices for products from an image, automating the process of cross-platform price comparison.

---

-   **Name**: Aekampreet Singh Dhir
-   **University**: Indian Institute of Technology Patna
-   **Department**: Engineering Physics

### Core Features

-   **Image-to-Query**: Utilizes a fine-tuned BLIP model to generate precise search queries from product images.
-   **Multi-Platform Scraping**: Scrapes top e-commerce sites (Amazon, Flipkart, Myntra) to find product listings.
-   **Intelligent Ranking**: Sorts results by the best price and relevance to the user's query.
-   **Interactive UI**: A simple and clean user interface built with Streamlit.

### Project Structure

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

bash
git clone [your-repository-url]
cd [repository-name]
pip install -r requirements.txt


# Fashion Product Price Comparison Agent

---

## 2. Download the Dataset
The model is fine-tuned on the **Fashion Product Images Dataset**.

Download it from the link in archive (2) following directory structure.

Place the dataset into a directory (e.g., `data/myntradataset/`) and ensure the image paths and descriptions are correctly formatted in a `.csv` file as expected by the fine-tuning script.

---

## 3. Run the Fine-Tuning Script
Execute the `run_finetuning.py` script to start training the model on the fashion dataset.  
This will generate model checkpoints in the `data/checkpoints` directory.

bash
python run_finetuning.py

## 4. Update the Agent to Use the Fine-Tuned Model
After fine-tuning is complete, update the agent to use the newly trained weights:

1. Open the `agent/core.py` file.  
2. Find the line where the `PriceComparisonAgent` is initialized.  
3. Change the parameter:

## 5. Launch the Streamlit App

Now you are ready to run the application! Use the following command in your terminal:

python -m streamlit run app.py


Your web browser should automatically open to the application's UI.

## 6. Test the Application

Image Search: Upload a product image and click "Search for Best Prices".

Text Search: Alternatively, enter a detailed product description in the text box.

The agent will display the generated search query and a table of the best prices found online.

## Development & AI Assistance

The ai_logs file is included in this repository to provide transparency into the development process.
It contains conversations with AI assistants that were instrumental in:

Debugging code

Structuring the agent's logic

Exploring implementation strategies


