# 🏦 USAA Fraud Research Project - Group 6 
### Alec Lundy • Jason Labrecque • Jonathan Corll • Maria Eduarda Cramer F De R Silva
### Source: Federal Deposit Insurance Corporation - FDIC

---

## Project Description

In partnership with the USAA Federal Savings Bank, our team from the DTSC 3602 course at UNC Charlotte was asked to develop a Python-based tool capable of:
   • Scraping fraud-relates articles from the FDIC
   • Cleaning and structuring unstandardized text 
   • Embedding articles into vectorspace 
   • Classifying fraud vs non-fraud content 
   • Clustering fraud types using semantic similarity 
   • Uploading standardized data into Supabase for downstream analytics

This project will contribute directly to USAA's State of Fraud quarterly report and internal weekly updates for fraud teams, allowing analysts to track trends, identify emerging fraud behaviors, and streamline fraud focused content discovery across multiple public sources. 

--- 

## ⚡ Quick Start

Use ``uv`` for fast inventory management and reproductible builds.  Follow these steps to run the full FDIC fraud-analysis pipeline o launch the dashboard:

*** 1. Clone the Repository ***

``` git clone https://github.com/alundy98/USDA-Fraud-Project ```

*** 2. Create and Activate the Environment ***
Create new uv-managed virtual environment:
``` bash
uv venv 
```
Activate it:
``` bash
source .venv/bin/activate
```

Install project dependencies from pyproject.toml and uv.lock:
``` bash 
uv sync
```

*** 3. Configure Environment Variables ***
This project uses:
   • Supabase for storing embeddings + article metadata 
   • OpenAI for embeddings and AI cleanup/summarization 

Create an .env file with:

```
SUPABSE_URL= 
SUPBASE_KEY= 
OPENAI_API_KEY=
```
This will be necessary for embedding generationa nd uploading the processed data to Supabase. 

*** 4. Run the Streamlit Dashboard ***
The dashboard allow you to explore:
   • scraped FDIC articles 
   • semantic clusters 
   • fraud pattern groupings 
   • embeddings 
   • similarity search 
   • model outputs 

Run it:
``` bash 
uv run streamlit_run.py
```

---

## Project Structure 

| File / Directory | Description |
|------------------|-------------|
| `data/` | Raw scraped FDIC articles |
| `outputs/` | Cleaned articles, embeddings, and clustering results |
| `outputs_not_fraud/` | Non-fraud reference articles used for comparison |
| `outputs_new_prompt/` | Experimental embedding outputs using alternative prompts |
| `FDIC_scraper.py` | Scraper for FDIC press releases and consumer alerts |
| `fdicOIG_scraper.py` | Scraper for FDIC OIG PDF reports |
| `embedding.py` | Main text cleaning and embedding pipeline |
| `embedding_with_ai.py` | Optional AI-assisted embedding variant |
| `detection_method_clustering.py` | Methods for clustering embeddings and exploring fraud patterns |
| `model_train.py` | Baseline classifier for fraud vs non-fraud detection |
| `supa_upsert.py` | Upload cleaned data + embeddings to Supabase |
| `streamlit_run.py` | Streamlit dashboard to explore articles and clusters |
| `example.env` | Template for required environment variables |
| `pyproject.toml` | Project configuration + dependencies (uv-managed) |
| `uv.lock` | Locked dependency versions for reproducibility |

---

## Methods Overview

*** Data Collection ***
We built two scrapers to extract FDIC content:
   • `FIDC_scraper.py` - scrapes FDIC search results, handling both HTML and PDF documents, and allowing custom queries 
   • `fdicOIG_scraper.py` - scrapes HTML releases from the FDIC OIG site
Each article will include title, publication date, text, and URL. 

*** Text Cleaning and Embedding ***
Articles are cleaned and converted into dense vector embeddings using the OpenAI API. Embeddings allow us to measure similarity and group related fraud cases together. After embedded, articles can then be compared, searched, and grouped.  

*** Supabase Integration ***
Processed articles and embeddings are uploaded to our Supabase database using: 

```supa_upsert.py```

This allowes our FDIC dataset to be merged with other teams' data sources in a centralized storage system if needed.  

*** Dashboard ***
The Streamlit dashboard `stramlit_run.py` allows used to interact with the FDIC dataset and review the results of our pipeline.  It includes:
   • an overview of all scraped FDIC articles 
   • details for each article such as title, date, URL, and full text 
   • visualizations generated from the processed dataset 
   • summaries of embedding outputs 
   • sorting and fultering options 
   • tables showing strcutrued metadata 
   • previews of fraud related content extracted during scraping 

The dashboard serves as a quick inspection of the dataset and confirmation that the scraping, cleaning, and embedding steps are runnign correctly before merging with other teams' sources.  

---

## Key Findings 

*** Clear fraud-related groupings in FDIC narratives ***
Our embedding-based analysis showed that FDIC articles were consistently organized around major fraud themes such as cuber fraud, identity theft, consumer deception, and payment-related fraud.  This can be seen as a pattern across years and provides us with a strong foundation for trend monitoring within fraud. 

*** Significant semantic shift between 2019 and 2020 ***
A substantial change in fraud narratives took place within 2019 and 2020, largely coinciding with the rise in digital activity during the COVID-19 pandemic.  This shift marks when cuber fraud began rapidly acceleating and ovrelapping with identity fraud. 

*** Increasing overlap between cyber fraud and identity fraud ***
Since 2020, cyber fraus and identity fraud have become semantically more similar.  This could potentially suggest that the two fraud types are converging in methods, tactics, or reporting patterns.  This may require joint monitoring in future models. 

*** Fraud narratives have recently stabilized ***
Compared to significantly sharp shifts before 2020, the last three yearss show a more stable fraud pattern.  This is a sign of both matured fraud tactics and more consistent reporting from the FDIC. 

*** Emphasis on education in FDIC reporting ***
Across all fraud categories, the FDIC strongly emphasized consumer educaiton.  Preventative messaging around personal information, cyber safety and awareness, and more, were frequent strategies described in our scraped articles.  

---

## Future Work 

   • Expand FDIC coverage and add more sources 
   • Enhance semantic trend monitoring 
   • Increase fraud-type granularity 
   • Strengthen classification models 
   • Improve dashboard interactivity 
   • Automate the pipeline for continuous monitoring 
   
---

You can access the FDIC Knowledge Center here: [FIDC Knowledge Center](https://ask.fdic.gov/fdicinformationandsupportcenter/s/article/Q-What-is-the-Grandparent-Scam?language=en_US)

This Python project scrapes articles from the FDIC website related to fraud, scams, and deceptive practices. It automatically extracts the article title, URL, full publication date, year, and the article text from both HTML pages and PDFs. Articles older than 10 years are automatically ignored. The results are saved to a CSV file (final.csv) for easy analysis or further processing. The scraper uses requests, BeautifulSoup, pdfplumber, and regular expressions to handle varied date formats and document types.


#To run on your own:
If you want to alter the query, just go to that same link*(the search page) and input your keywords or filter(-inactive, etc..) and then put that
link as the base_url, if it breaks double check that the xpath and css are exactly correct to the path in the html.
Ignore the js
None of the Information and support center links work, the search bar and containers mess with the scraper. if Needed, we can write a helper 
explicitly for q and a pages, though I think thats not necessary.
Also, several links were already dead before scraping, so those are removed

No more Selenium-> the js loading wasn't actually in the way of what we needed to get, so ignoring it entirely
and just directly putting the xpath or css for the html items we needed worked better, but slower.
No API: I got an api key, but I guess it was access to their bank records and stuff like that, couldn't get it to accept my key
and scrape the articles. Might be possible and I just didn't do it right, but I tried for ages and got nowhere.

Supabase Schema:
fdic_articles = the orignal cleaned full text, year, date, and url of our articels
non_fraud_articles = a collection of articles from fdic in the same time span that are not fraud related
embedding_cluster_summary = a dataset showing info about the 4 clusters
full_article_embeddings = holds url, relevance score, and cluster they belong to
article_embedding_full = the meat, full embeddings values for every article with text, url, and cluster

Embedding style using OpenAI:
Python pipeline for processing non-fraud articles. Uses Supabase to fetch data. Text cleaning uses re, string, NLTK for stopwords and lemmatization. Embeddings generated with OpenAI GPT-4o-mini, chunked for long texts. Summaries and fraud-type/detection labels also via GPT API. Data handled with pandas and numpy. Clustering done with HDBSCAN, fallback to KMeans if too few clusters. TF-IDF extracts keywords per cluster. Cosine similarity computes article relevance to cluster centroids. Outputs include CSV and Parquet files with embeddings, clusters, keywords, summaries, and relevance scores. Handles large datasets but long texts may fail embedding. Computationally intensive.

Model training in model_train.py:
Python pipeline for training fraud detection models. Uses Supabase to fetch embeddings and labels. Data handled with pandas and numpy. Embeddings parsed safely with ast and json. Binary fraud detection uses XGBoost with train/test split from scikit-learn. Fraud type classification uses LabelEncoder and multi-class XGBoost, filtering out rare fraud types. Models and encoder saved with joblib. Environment variables loaded with dotenv. Successfully trains and saves both binary and multi-class models. Handles missing or malformed embeddings. Computationally efficient but depends on precomputed embeddings.