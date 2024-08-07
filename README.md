# Automatic-Write-Off-Detection

## Overview
Welcome to the Automatic Write-Off Detection project! This project is inspired by the article from Keeper Tax on creating an efficient system to identify tax write-offs automatically. This guide will walk you through setting up the project, understanding its components, and running it on your machine.

## Project Goals
1. Identify merchants from transaction descriptions
2. Categorize transactions into various categories like Purchase, Debit, Payment, etc.
3. Detect and handle duplicate transactions

## Features
- **Merchant Identification**: Uses Regex and string processing techniques to identify merchants from transaction descriptions
- **Transaction Categorization**: Classifies transactions using a machine learning model
- **Duplicate Handling**: Detects and handles duplicate transactions

## Setup

### Prerequisites
Before you start, ensure you have the following installed on your machine:
- Python 3.x
- Git

### Step-by-Step Setup
1. **Clone the Repository**
   To clone this repository to your local machine, in the terminal run:
   ```bash
   git clone https://github.com/yourusername/automatic-write-off-detection.git
   cd automatic-write-off-detection
2. **Set Up a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
3. **Install Required Libraries**
   The necessary Python libraries are in the requirements.txt file:
   ```bash
   pip install -r requirements.txt

## Running the Project
- set your OpenAI API key in the code: openai.api_key = 'your_openai_api_key' (replace 'your_openai_api_key' with your actual API key) in pipeline.py

### Data Preparation
   I created an example CSV file with transaction data in the data directory. That file is necessary for the scripts to run.
   
### Executing the Main Script
   To run the main part of the project which processes the data, simply execute the main.py script:
   ```bash
      python src/main.py 
   ```
## Data and Model Notebooks
- data_exploration.ipynb: Explore and understand transaction data
pipeline and model training

## Pipeline and Model Training
### Preprocess Data
- preprocess_data function in pipeline.py cleans and prepares the data for model training

### Train the Model
- model training and pipeline setup are managed within pipeline.py, which trains a Naive Bayes or Linear Regression model (whichever is specified) and integrates a GPT-4/ChatGPT model to enhance extraction

### Evaluate the Model
- model is evaluated for accuracy, precision, recall, and other metrics and  results are printed and saved in the models directory

### Expanded Data Example
- This data was created from Keeper Tax's transaction data for a fictitious Chicago-based freelance graphic designer
```bash
date,plaid_merchant_description,amount,plaid_category,write_off_status (basic rules),ground_truth,keeper_merchant_description,keeper_category,write_off_status (keeper first pass),write_off_status (keeper after 1st session)
2022-01-02,Zelle Transfer Conf# nvizxvsui4; Tim,-1960.00,"Transfer, Credit",no,no,Zelle Transfer - Tim,↔️ transfer,no,no
2022-01-02,TRELLO.COM* ATLASSIAN XXXXXXXXXXXX0497,14.99,"Shops, Digital Purchase",needs review,yes,Trello,💻 software,yes,yes
2022-01-03,MOBILE PURCHASE 01/03 MARATHON PETRO11,73.52,"Travel, Gas Stations",no,no,Marathon,⛽ gas fill up,no,no
2022-01-03,PGANDE DES:WEB ONLINE ID:XXXXXXXXXX8324 INDN:JOHNNY CO ID:XXXXX11632 WEB,90.15,Utilities,no,yes,PG&E,🏠 utilities,yes,yes
2022-01-04,PMNT SENT 0104 APPLE CASH - SENT CA XXXXX3630XXXXXXXXXX8324,744.20,"Payment, Credit Card",no,no,Apple Card Payment,↔️ transfer,no,no
