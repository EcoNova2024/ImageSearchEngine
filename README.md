# Image Search Engine

This project implements an image search engine using a pre-trained ResNet-18 model for feature extraction. The search engine can retrieve similar images based on a query image or image URL.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Features

- Search for similar images using a query image or a URL.
- Efficient image feature extraction using deep learning.
- Cosine similarity to find and rank similar images.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Feature Extraction**: Run the following command to extract features from images in a specified directory:

   ```bash
   python train.py
   ```

2. **Start the Flask Server**: Run the Flask app to start the server:

   ```bash
   python app.py
   ```

3. **Make Requests**: Use a tool like `curl`, Postman, or your preferred method to make requests to the API.

## API Endpoints

### Search for Similar Images

- **URL**: `/search`
- **Method**: `POST`
- **Request Body**:

  - To send an image file:
    ```json
    {
      "image": "<file>"
    }
    ```
  - To send an image URL:
    ```json
    {
      "url": "<image-url>"
    }
    ```

- **Response**:
  - Returns a JSON array of similar images with their names and similarity scores:
    ```json
    [
      {
        "name": "image_name_1.jpg",
        "similarity": 0.85
      },
      {
        "name": "image_name_2.jpg",
        "similarity": 0.8
      }
    ]
    ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bugs.
