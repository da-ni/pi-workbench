## Linux Language Model (LLM)
This repository contains code for training and using a language model for Linux-related text. The model is based on OpenAI's GPT-3 and is fine-tuned on the Linux Bible.

### Setting up a Virtual Environment
To ensure that your project's dependencies are isolated from other Python projects on your system, it's a good practice to set up a virtual environment. To set up a virtual environment using venv, follow these steps:

1. Open a terminal or command prompt and navigate to the root directory of the project.

2. Run the following command to create a new virtual environment:
 `python -m venv env`

This will create a new directory called env in the current directory, containing the virtual environment. (repo is tested with version python3.11)

3. Activate the virtual environment by running the following command:
`source env/bin/activate` 

This will activate the virtual environment and change your shell's prompt to indicate that you're now using the virtual environment.

### Installation
To install the dependencies for this project, you can use pip to install the packages listed in the requirements.txt file:

`pip install -r requirements.txt`

### Setting up the OpenAI API Key
To use the Linux Language Model, you will need to obtain an API key from OpenAI. You can sign up for an API key on the [OpenAI website](https://platform.openai.com/overview).

Once you have obtained an API key, you can set it as an environment variable in a .env file in the root directory of the project: 
- Replace the placeholder in the `example.env` file with your openai api key
- Then, rename the `example.env` file to `.env`.

### Creating a Vector Database
To create a vector database for the Linux Language Model, you can use the `vectorize.py` script. This script reads in documents (mutliple types supported) and creates a vector representation of each document using the penai embeddings model. The vectors are then stored in a database for efficient retrieval.

To create a vector database, you can run the following command:

`python create_db.py [--source-dir /path/to/text/files] [--output-dir /path/to/vector/db]`

This will create a vector database in the specified output directory, containing vector representations of all the text files in the source directory.

### Running the LLM
To run the Linux Language Model, you can use the linus.py script. This script loads the vector database and the language model, and provides an interactive terminal session for prompting the model.

To run the LLM, you can run the following command:

`python linus.py [--db-dir /path/to/vector/db]`

### Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome contributions of all kinds, including bug fixes, new features, and improvements to the documentation.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

###TEST
