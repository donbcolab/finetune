# finetune

### Application Architecture

| File Name         | File Type | Description                                                                                                                           |
|-------------------|-----------|---------------------------------------------------------------------------------------------------------------------------------------|
| `app.py`          | Python    | The main Gradio application file that defines the user interface, handles user interactions, and orchestrates the fine-tuning process. |
| `utils.py`        | Python    | Contains utility functions for model loading, formatting, and error handling.                                                        |
| `fine_tuning.py`  | Python    | Implements different fine-tuning methods (SFT, DPO, PPO) as subclasses of `FineTuningMethod`, including training and evaluation logic. |
| `evaluation.py`   | Python    | Defines evaluation metrics (Accuracy, Perplexity, ROUGE) as subclasses of `EvaluationMetric` for assessing model performance.        |
| `datasets.py`     | Python    | Handles dataset loading, preparation, tokenization, and collation.                                                                     |
| `config.json`     | JSON      | Configuration file that stores available datasets, fine-tuning methods, evaluation metrics, and other app settings.                    |
| `requirements.txt`| Text      | Lists the required Python dependencies for running the application.                                                                    |

### Python Code Files

**1. `app.py`:**

*   **Imports:**
    *   `os`, `json`, `logging`, `gradio as gr`
    *   `utils`, `fine_tuning`, `evaluation`, `datasets`
*   **Classes:** None
*   **Methods:**
    *   `handle_submit(model_name)`: Loads and displays the model architecture.
    *   `handle_fine_tune(model_name, dataset_file, dataset_hub, method, tokenization, collation, lora_r, lora_alpha, lora_dropout, reward_model_name, num_train_epochs, per_device_train_batch_size, gradient_accumulation_steps, **training_kwargs)`: Fine-tunes the selected model using the chosen method and parameters.
    *   `handle_evaluate(model_name, metric)`: Evaluates the fine-tuned model using the selected metric.

**2. `utils.py`:**

*   **Imports:**
    *   `subprocess`, `os`, `torch`, `transformers`, `spaces`, `joblib` (for caching)
*   **Classes:** None
*   **Methods:**
    *   `install_flash_attn()`: Installs the `flash-attn` library.
    *   `get_model_summary(model_name)`: Retrieves the model summary, including handling 4-bit quantization. 
    *   `get_available_options(key, config)`: Retrieves available options from `config.json`.
    *   `format_metrics_output(metrics)`: Formats the metrics dictionary for display.

**3. `fine_tuning.py`:**

*   **Imports:**
    *   `transformers`, `trl`, `utils`, `evaluation`, `peft` (for LoRA)
*   **Classes:**
    *   `FineTuningMethod`: Abstract base class for fine-tuning methods.
    *   `SFTMethod(FineTuningMethod)`: Implements Supervised Fine-tuning (SFT).
    *   `DPOMethod(FineTuningMethod)`: Implements Direct Preference Optimization (DPO).
    *   `PPOMethod(FineTuningMethod)`: Implements Proximal Policy Optimization (PPO).
*   **Methods:**
    *   `get_fine_tuning_method(method_name, config)`: Returns the appropriate fine-tuning method class based on the selected method from the UI.

**4. `evaluation.py`:**

*   **Imports:**
    *   `transformers`, `datasets`
*   **Classes:**
    *   `EvaluationMetric`: Abstract base class for evaluation metrics.
    *   `AccuracyMetric(EvaluationMetric)`: Calculates accuracy.
    *   `PerplexityMetric(EvaluationMetric)`: Calculates perplexity.
    *   `RougeMetric(EvaluationMetric)`: Calculates ROUGE scores.
*   **Methods:**
    *   `get_evaluation_metric(metric_name, config)`: Returns the appropriate evaluation metric class based on the selected metric from the UI.

**5. `datasets.py`:**

*   **Imports:**
    *   `datasets`, `pandas as pd`, `transformers`, `joblib` (for caching)
*   **Classes:** None
*   **Methods:**
    *   `load_and_prepare_dataset(dataset_name, config, source='hub')`: Loads and prepares the selected dataset, handling both Hub datasets and local CSV files.
    *   `tokenize_and_collate(dataset, model_name, tokenization, collation, config, **kwargs)`: Tokenizes and collates the dataset based on selected strategies, including custom options.