
# Book Recommendation System

A Hybrid recommendation engine that mixes <br>**Content-Based Filtering (CBF)** + **Collaborative Filtering (CF)** <br> to serve up book picks that actually match your vibe. <br> 
- Trained on 10K Goodreads titles + 6M ratings so it knows what’s good.
- 🔍 For bookworms, Powered by Python, scikit-learn<br> for ML newbies
---

## System Snapshot 📷

![Image](https://i.ibb.co/GysbVKp/Screenshot-2025-09-19-233947.png)


## Acknowledgements      
- Thanks [QWEN_AI](https://chat.qwen.ai/) for assisting with the Streamlit GUI integration.
        
    
        

## Features

- ✅ **Hybrid Model**: 70% content-based + 30% collaborative (adjustable)
- 🔍 **Fuzzy Search**: Handles typos & partial titles via `rapidfuzz`
- 📚 **Author Fallback**: Search by author if title not found
- 🚀 **Fast Load**: Precomputed matrices
- 🖼️ **Interactive UI**: Streamlit grid with Book covers, ratings, authors
 

---

## Techniques

- **Content-Based**: TF-IDF on tags/authors → cosine similarity
- **Collaborative**: Truncated SVD on user ratings → similarity
- **Hybrid**: Weighted average of both scores
- **Fuzzy Matching**: RapidFuzz for robust search
---

## Dataset

📊 **Source**: [Goodreads 10K Dataset on Kaggle](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)  
📁 Includes:  
- `books.csv` (10K books)  
- `ratings.csv` (~6M ratings)  
- `tags.csv`, `book_tags.csv`, `to_read.csv`
---

## Requirements

```bash
pip install -r requirements.txt
```
### Important⚠️:
 Download the datset as ZIP by clicking here before running the noteboook 
[goodreads_dataset ](https://storage.googleapis.com/kaggle-data-sets/1938/3914/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250919%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250919T212410Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=a7790bbf5dfc281a9319bf614f3d14b41694c513e79be12354c1327b581fcc09b1a9cdf46bbc204c4386a69705ba18e32d61797db3a255ad519c7751f9c93c71ba1984f265922ad1b5a69bf5c90163e388a8e8d2e8bd6ccb050d9668bf0befdfabc528fa3335476b7d9cfc47e001f86990df6184af115d123e420a247cda2a92ff90f0ae393fd71f01279f74656a690f54052e58ad05140bee435a73bf3410bacf555ae057095c0642d2cd14579110e1f12d25b146a3035f9d40fc08527689d22300d5c0524fcad5d28692edf6a40313118d855ed30e46257e192f48f97c344c5b333d3ad06b541f9eb5983640766c93ddce860eaad7d809a5dd1601154b439d)
        
---

## Setup & Run

1. Clone repo:
```bash
git clone https://github.com/Mohamedashraf2005/Book_RecommendationSystem.git
cd Book_RecommendationSystem
```


2. Install deps & run notebook:
```bash
pip install -r requirements.txt
jupyter notebook BooksRecommendationSystem.ipynb
```
### Important⚠️: 
Run all cells in <br> <br>**BooksRecommendationSystem.ipynb** <br >
<br> This trains/prepares the hybrid model and saves the required .joblib files.
<br>
3. Launch app:
```bash
streamlit run BookRecommendationSystem_ST.py
```
→ Visit `http://localhost:8501`

---

## Usage

1. Type a Book title or Even Tags→ get fuzzy suggestions
2. Pick one (Add an author for fallback)
3. Adjust sliders in sidebar for <br> 
```
Number of Recommendations, System Weight (0.7 Recommended)
```
4. Click “View Recommended Books”
---


## Contact

- [mohamedachrvf@gmail.com](mailto:mohamedachrvf@gmail.com)
---
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
