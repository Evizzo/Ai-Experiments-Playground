# OpenAI Fine-Tuning Script

This script provides a command-line interface to automate the process of fine-tuning an OpenAI model. 
It handles uploading the training data, creating the fine-tuning job, and monitoring its progress until completion.
It also calculates how much fine-tuning will cost you, and validates fine-tuning data.

Goal is to make an AI act as he knows he is just an AI, inspired by show Mr. Robot.

Training data is created with ChatGPT o4 mini high.

## How It Works

The script performs the following steps in sequence:

1.  **Initialize Client**: It loads your OpenAI API key from an environment file (`.env`).
2.  **Upload File**: It uploads your local training data file (in `.jsonl` format) to OpenAI's servers.
3.  **Create Job**: It initiates a new fine-tuning job using the uploaded file and the specified base model.
4.  **Monitor Status**: It periodically checks the status of the job and logs the progress to the console.
5.  **Report Result**: Once the job is complete, it logs the final status and the name of the new, fine-tuned model.

## Prerequisites

Before running the script, ensure you have the following set up:

1.  **Python**: Python 3.7 or newer must be installed.
2.  **Libraries**: Install the required Python libraries. It's recommended to do this in a virtual environment.

    ```bash
    pip install openai python-dotenv tiktoken
    ```

3.  **API Key**: Create a file named `.env` in the same directory as the script. Add your OpenAI API key to this file:

    ```
    OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

4.  **Training Data**: Your training data must be in a JSONL file, where each line is a valid JSON object 
representing a single training example.

## How to Run

To start the fine-tuning process, you must now specify the `create` action. The script accepts the action, 
the path to your training file, and the name of the base model.

**Command:**

```bash
python fineTuning.py create [TRAINING_FILE_PATH] [MODEL_NAME] [--epochs NUMBER]
```

**Example:**

To train the `gpt-4.1-nano-2025-04-14` model with the data in `escapedLlmDataset.jsonl`, use this command:

```bash
python fineTuning.py escapedLlmDataset.jsonl gpt-4.1-nano-2025-04-14 --epochs 5
```

Default epochs is 3.

The script will then begin the process and log its progress in the terminal.