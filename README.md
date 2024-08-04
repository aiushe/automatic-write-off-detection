# Automatic-Write-Off-Detection

## Overview
Welcome to the Automatic Write-Off Detection project! This project is inspired by the article from Keeper Tax on creating an efficient system to identify tax write-offs automatically. This guide will walk you through setting up the project, understanding its components, and running it on your machine.

## Project Goals
1. Identify merchants from transaction descriptions
2. Categorize transactions into various categories like Purchase, Debit, Payment, etc.
3. Detect and handle duplicate transactions

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