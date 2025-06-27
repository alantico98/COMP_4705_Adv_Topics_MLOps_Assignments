# Movie Review Sentiment Analyzer

This app uses a Naive Bayes classifier to analyze movie reviews and predict whether they are **positive** or **negative**.

## How to Run

Follow these steps to run the app locally:

1. Clone the repo using bash (if on Windows, use git-bash or anything similar that allows for ssh cloning):
    bash
    git clone git@github.com:alantico98/COMP_4705_Adv_Topics_MLOps_Assignments.githttps://github.com/yourusername/imdb_sentiment_app.git
    cd ./COMP_4705_Adv_Topics_MLOps_Assignments/Assignment_1 # The overall repo stores all the assignments, but each assignment folder contains the assignments specific files

2. (Optional but recommended) Create a virtual environment
    python -m venv .venv
    source .venv\Scripts\activate # If using Windows
    source .venv/Scripts/activate # If using Git-bash on Windows
    source .venv/bin/activate     # If using Linux

3. Install dependencies
    pip install -r requirements.txt

4. Download IMDB Dataset and rename
    - https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    - Move the csv file to the git repo Assignment_1 folder (./COMP_4705_Adv_Topics_MLOps_Assignments/Assignment_1)
    - mv ./COMP_4705_Adv_Topics_MLOps_Assignments/Assignment_1/'IMDB Dataset.csv' ./COMP_4705_Adv_Topics_MLOps_Assignments/Assignment_1/imdb_dataset.csv

5. Run the training script once
    python train_model.py

6. Start the Streamlit app
    streamlit run app.py

# Note: imdb_dataset.csv and streamlit_labels.pkl were added to .gitignore due to their size being too big