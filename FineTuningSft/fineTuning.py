import json
import os
import time
import logging
import argparse
import tiktoken
from openai import OpenAI, APIError
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

POLL_INTERVAL_SECONDS = 30
JOB_TIMEOUT_SECONDS = 3600


def createOpenaiClient() -> OpenAI:
    """Load the API key and create an OpenAI client."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY is not set in a .env file or environment variables.")
        raise ValueError("Missing OpenAI API key.")
    return OpenAI(api_key=api_key)


def uploadTrainingFile(client: OpenAI, file_path: str) -> str:
    """Upload a JSONL file for fine-tuning and return its ID."""
    if not os.path.exists(file_path):
        logging.error(f"Training file not found at path: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    logging.info(f"Uploading training file: {file_path}")
    try:
        with open(file_path, "rb") as f:
            response = client.files.create(file=f, purpose="fine-tune")
        logging.info(f"File uploaded successfully. File ID: {response.id}")
        return response.id
    except APIError as e:
        logging.error(f"An API error occurred while uploading the file: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise


def createFineTuningJob(client: OpenAI, training_file_id: str, model_name: str, n_epochs: int = 'auto') -> str:
    """Start a fine-tuning job with optional hyperparameters."""
    logging.info(f"Creating a fine-tuning job for model: {model_name} with {n_epochs} epochs.")
    try:
        hyperparameters = {
            "n_epochs": n_epochs,
            # "batch_size": 1,
            # "learning_rate_multiplier": 1
        }
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=model_name,
            hyperparameters=hyperparameters
        )
        logging.info(f"Fine-tuning job created successfully. Job ID: {job.id}")
        return job.id
    except APIError as e:
        logging.error(f"An API error occurred while creating the fine-tuning job: {e}")
        raise


def monitorFineTuningJobStatus(client: OpenAI, job_id: str) -> str | None:
    """Monitor the fine-tuning job's status until completion or timeout."""
    start_time = time.time()

    while time.time() - start_time < JOB_TIMEOUT_SECONDS:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            logging.info(f"Job status: {status}")

            if status in ("succeeded", "failed", "cancelled"):
                logging.info("Retrieving recent events:")
                events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
                for event in reversed(events.data):
                    logging.info(f"- {event.message}")
                return job.fine_tuned_model

            time.sleep(POLL_INTERVAL_SECONDS)

        except APIError as e:
            logging.error(f"An API error occurred while checking job status: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)

    logging.error("Job monitoring timed out. Aborting.")
    return None

def validateTrainingFile(file_path: str) -> bool:
    """Reads a JSONL file and validates its format for OpenAI fine-tuning."""
    logging.info(f"Validating data format for {file_path}...")
    errors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                if 'messages' not in obj or not isinstance(obj['messages'], list):
                    errors.append(f"Line {line_num}: Missing or invalid 'messages' key.")
                    continue

                if len(obj['messages']) < 2:
                    errors.append(
                        f"Line {line_num}: 'messages' array should have at least two messages (user, assistant).")

                for msg in obj['messages']:
                    if 'role' not in msg or 'content' not in msg:
                        errors.append(f"Line {line_num}: A message is missing 'role' or 'content'.")
            except json.JSONDecodeError:
                errors.append(f"Line {line_num}: Not a valid JSON object.")

    if errors:
        for error in errors:
            logging.error(error)
        return False

    logging.info("Data validation successful.")
    return True


def estimateTuningCost(file_path: str, model_name: str, epochs: int) -> float:
    """Estimates the cost of a fine-tuning job."""
    pricing_per_1k_tokens = 0.0015 # this is for gpt-4.1-nano-2025-04-14

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for message in obj['messages']:
                total_tokens += len(encoding.encode(message['content']))

    estimated_cost = (total_tokens / 1000) * pricing_per_1k_tokens * epochs
    logging.info(f"Total tokens in dataset: {total_tokens}")
    logging.info(f"Training for {epochs} epochs.")
    logging.info(f"Estimated cost: ${estimated_cost:.4f} USD")
    return estimated_cost


def main():
    """Main function to run the entire fine-tuning process."""
    parser = argparse.ArgumentParser(description="A script to fine-tune an OpenAI model.")
    parser.add_argument("training_file", help="Path to the JSONL training file.")
    parser.add_argument("model_name", help="The base model to fine-tune (e.g., 'gpt-3.5-turbo').")
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs to run."
    )
    args = parser.parse_args()

    if not validateTrainingFile(args.training_file):
        logging.error("Aborting due to data validation errors.")
        return

    logging.info("Data validation passed.")

    estimated_cost = estimateTuningCost(args.training_file, args.model_name, args.epochs)
    proceed = input(f"Estimated cost is ${estimated_cost:.4f}. Proceed? (y/n): ")
    if proceed.lower() != 'y':
        logging.info("Aborting job.")
        return

    try:
        client = createOpenaiClient()
        training_file_id = uploadTrainingFile(client, args.training_file)
        job_id = createFineTuningJob(client, training_file_id, args.model_name, n_epochs=args.epochs)
        fine_tuned_model_name = monitorFineTuningJobStatus(client, job_id)

        if fine_tuned_model_name:
            logging.info("Fine-tuning process completed successfully.")
            logging.info(f"The name of your new fine-tuned model is: {fine_tuned_model_name}")
        else:
            logging.warning("The fine-tuning process did not complete successfully.")

    except (ValueError, FileNotFoundError, APIError) as e:
        logging.error(f"Process terminated due to an error: {e}")
    except Exception as e:
        logging.error(f"An unexpected critical error occurred: {e}")


if __name__ == "__main__":
    main()