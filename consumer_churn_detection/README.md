# Consumer Churn Detection

This project detects consumer churn for a subscription-based service or business using a machine learning model.

## Setup

1. Install the necessary libraries:

   ```sh
   pip install -r requirements.txt
   ```

2. Train the model:

   ```sh
   python scripts/train.py
   ```

3. Evaluate the model:

   ```sh
   python scripts/evaluate.py
   ```

4. Run the Streamlit app:

   ```sh
   streamlit run app/main.py
   ```

## File Structure

- `data/`: Contains the dataset file.
- `models/`: Contains the trained model and scaler.
- `scripts/`: Contains the training and evaluation scripts.
- `app/`: Contains the Streamlit web app.
- `requirements.txt`: Lists the required Python libraries.
- `README.md`: Project documentation.
