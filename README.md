# Drug Recommendation System

- **App**: [Disease Condition and Drug Recommendation](https://passive-aggclass-drug-recommend.streamlit.app/)

This project is a Machine Learning-based Drug Recommendation System. The application uses the Passive-Aggressive Classifier model to predict and recommend drugs based on user input. The project includes a Streamlit web application that serves the machine learning model.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Contact](#contact)

## Project Overview

This project aims to provide drug recommendations based on a machine learning model trained on drug review datasets. The Streamlit app interacts with users, taking input to predict the most suitable drug and displaying the results in an easy-to-understand format.

## Features

- Drug recommendation based on user input.
- Machine learning model built using the Passive-Aggressive Classifier.
- Streamlit-based web application for easy interaction.
- Models deployed on Hugging Face for portability and easy access.
- Containerized using Docker for easy deployment.

## Installation

To install and run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abdul-Khadar-Jilani/Drug_AI.git
   cd Drug_AI
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage
Once the application is running, you can access it via the URL provided by Streamlit in your terminal (e.g., http://localhost:8501/). Enter the required input, and the application will provide drug recommendations based on the trained machine learning model.

## Technologies Used
- **Python**: Core language used for the machine learning model and Streamlit web framework.
- **Streamlit**: Web framework for serving the application.
- **Scikit-learn**: Machine learning library used for training the Passive-Aggressive Classifier.
- **Hugging Face**: Models are deployed in the Hugging Face repository for easy access and sharing.
- **Docker**: Containerization of the application for easy deployment.
- **Streamlit Cloud**: Deployment platform for the Streamlit application.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request to contribute.

## Contact
For any questions or suggestions, feel free to contact me:

- **Name**: Abdul Khadar Jilani
- **Email**: [jilaniabdulgec@gmail.com](mailto:jilaniabdulgec@gmail.com)
- **LinkedIn**: [https://www.linkedin.com/in/abdulkhadarjilani/](https://www.linkedin.com/in/abdulkhadarjilani/)
