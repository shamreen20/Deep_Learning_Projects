# Next Word Prediction Using LSTM

## Project Overview

This project develops a deep learning model to predict the next word in a sequence of words using Long Short-Term Memory (LSTM) networks. The model is trained on the text of Shakespeare's Hamlet, leveraging its complex language to create a robust next-word prediction system. The project encompasses the following steps:

1. **Data Collection:** The dataset consists of the text from Shakespeare's Hamlet, providing a rich and challenging corpus for sequence prediction.

2. **Data Preprocessing:** The text is tokenized, converted into sequences, and padded to ensure uniform input lengths. The data is split into training and testing sets.

3. **Model Building:** The LSTM model includes an embedding layer, two LSTM layers, and a dense output layer with softmax activation to predict the probability of the next word.

4. **Model Training:** The model is trained with early stopping to prevent overfitting, monitoring validation loss to halt training when performance plateaus.

5. **Model Evaluation:** The model's performance is tested using example sentences to assess its ability to accurately predict the next word.

6. **Deployment:** A Streamlit web application enables users to input a sequence of words and receive real-time predictions for the next word.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository:**
   """
   git clone https://github.com/your-username/next-word-prediction-lstm.git
   cd next-word-prediction-lstm
   """
2. **Create a Virtual Environment (optional but recommended):**
   """
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   """
3. **Install Dependencies:** Install the required Python packages using the provided `requirements.txt`:
   """
   pip install -r requirements.txt
   """
   The requirements.txt includes:
   - tensorflow (for LSTM model)
   - numpy (for numerical operations)
   - streamlit (for web app deployment)
   - Other dependencies as needed

4. **Download the Dataset:** The project uses the text of Hamlet. Ensure the dataset `(hamlet.txt)` is placed in the `data/` directory. You can download it from a public source  or use the provided sample data.

## Usage
1. Train the Model: Run the training script to preprocess the data and train the LSTM model:
"""
python train_model.py
"""
The script will:
- Load and preprocess the Hamlet text
- Build and train the LSTM model
- Save the trained model to the models/ directory

2. Run the Streamlit App: Launch the Streamlit web application to interact with the model:
"""
streamlit run app.py
"""
- Open your browser and navigate to http://localhost:8501
- Input a sequence of words, and the app will display the predicted next word.

3. Evaluate the Model: Use the evaluation script to test the model on example sentences:
"""
python evaluate_model.py
"""
## File Structure
"""
next-word-prediction-lstm/
├── data/
│   └── hamlet.txt              # Dataset: Shakespeare's Hamlet text
├── models/
│   └── lstm_model.h5           # Trained LSTM model
├── train_model.py              # Script for data preprocessing and model training
├── evaluate_model.py           # Script for model evaluation
├── app.py                      # Streamlit web app for real-time predictions
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
"""


## Dependencies

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Streamlit
- (Additional dependencies listed in requirements.txt)

## Notes

- The model performance depends on the quality and size of the training data. Hamlet provides a good starting point, but larger datasets or additional preprocessing may improve results.

- Early stopping is configured to monitor validation loss with a patience of 5 epochs to prevent overfitting.

- The Streamlit app requires an active internet connection for the initial setup but runs locally afterward.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/your-feature)
3. Commit your changes (git commit -m 'Add your feature')
4. Push to the branch (git push origin feature/your-feature)
5. Open a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

##Contact

For questions or suggestions, please open an issue on the GitHub repository or contact [shamreen.tabassum@mailbox.tu-dresden.de].