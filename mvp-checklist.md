# mvp checklist

**Critical Checklist based on the Notebook:**

1.  **4-bit Quantization:**  **DONE**
2.  **PEFT (LoRA):**  **DONE**
3.  **Dataset from CSV:**  **DONE**
4.  **Custom Tokenization:**  **PARTIALLY DONE** - Placeholder in `datasets.py`; example provided.
5.  **Custom Collation:**  **PARTIALLY DONE** - Placeholder in `datasets.py`; example provided.
6.  **Training with `transformers.Trainer`:** **DONE**
7.  **ROUGE Evaluation:  **DONE**
8.  **Saving Model & Tokenizer:** **DONE**

**Action Items:**

*   **Complete Custom Logic Placeholders:** have working examples of custom tokenization and collation, but need to implement the specific logic based on user input in  `datasets.py`.
*   **Add Training Argument Components to UI:**  Provide UI elements in  `app.py` for additional training arguments like  `num_train_epochs`, `per_device_train_batch_size`, and  `gradient_accumulation_steps`.  These components are already in the UI, but need to pass their values to the fine-tuning methods. 

### 1. Completing Custom Logic in `datasets.py`

```python
from datasets import load_dataset, load_from_disk
import pandas as pd
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DefaultDataCollator
from joblib import Memory

# ... (Existing caching setup)

def tokenize_and_collate(dataset, model_name, tokenization, collation, config, **kwargs):
    # ... (Existing code)

    elif tokenization == "custom":
        def tokenize_function(examples):
            # Get max length values from config
            max_input_length = config["tokenization_strategies"]["custom"]["max_input_length"]
            max_output_length = config["tokenization_strategies"]["custom"]["max_output_length"]

            # Tokenize the prompts and completions separately 
            inputs = tokenizer(
                examples["prompt"], 
                padding="max_length", 
                truncation=True, 
                max_length=max_input_length 
            )
            outputs = tokenizer(
                examples["completion"], 
                padding="max_length", 
                truncation=True, 
                max_length=max_output_length 
            )

            # Concatenate inputs and outputs
            input_ids = inputs["input_ids"] + outputs["input_ids"]
            attention_mask = inputs["attention_mask"] + outputs["attention_mask"]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": outputs["input_ids"],
            }

    # ... (Existing code)

    elif collation == "custom":
        def data_collator(features):
            # Custom collation logic based on the notebook
            input_ids = [feature["input_ids"] for feature in features]
            attention_mask = [feature["attention_mask"] for feature in features]
            labels = [feature["labels"] for feature in features]

            # Pad to the maximum length in the batch
            max_length = max(len(ids) for ids in input_ids)
            padded_input_ids = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
            padded_attention_mask = [mask + [0] * (max_length - len(mask)) for mask in attention_mask]
            padded_labels = [labs + [-100] * (max_length - len(labs)) for labs in labels]

            return {
                "input_ids": torch.tensor(padded_input_ids),
                "attention_mask": torch.tensor(padded_attention_mask),
                "labels": torch.tensor(padded_labels),
            }
    # ... (Existing code)
```

**Explanation:**

*   **`tokenize_function`:**  This now mirrors the notebook's approach.  It gets the `max_input_length` and `max_output_length` from the `config.json` (make sure to add these).  It tokenizes "prompt" and "completion" separately, concatenates them, and returns the correct dictionary format for the  `Trainer`.
*   **`data_collator`:**  This function now performs padding to the maximum length within the batch, aligning with the notebook.  This logic can be adapted if a different collation strategy is needed.

### 2. Passing Training Arguments from `app.py`

```python
# ... (Existing imports)

# ... (Gradio interface setup)

    def handle_fine_tune(model_name, dataset_file, dataset_hub, method, tokenization, collation, lora_r, lora_alpha, lora_dropout, reward_model_name, num_train_epochs, per_device_train_batch_size, gradient_accumulation_steps, **training_kwargs):
        try: 
            # ... (Existing dataset loading logic)

            # Get hyperparameters for the selected method
            selected_hyperparams = {
                k[1]: v for k, v in training_kwargs.items() if k[0] == method
            }

            # Get training arguments from Gradio components
            training_args_dict = {
                "output_dir": CONFIG["fine_tuning_methods"][method]["output_dir"],
                "num_train_epochs": num_train_epochs,
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": training_kwargs[(method, "learning_rate")],
                # ... Add other training arguments from training_kwargs
            }

            # ... (Rest of the handle_fine_tune function)
```

**Explanation:**

*   The `handle_fine_tune` function now gathers values from the  `num_train_epochs`,  `per_device_train_batch_size`, and  `gradient_accumulation_steps` components. 
*   These values are used to create a  `training_args_dict`  which is passed to the  `FineTuningMethod.train()`  method, allowing the  `Trainer`  to use these settings.

### Completing the MVP:

1.  **Update UI:**  Add UI elements for the new  `max_input_length`  and  `max_output_length`  parameters in the "Custom" tokenization tab of the Gradio interface.
2.  **Update `config.json`:** Add the `max_input_length` and `max_output_length` with default values to the "custom" tokenization section.
3.  **Thorough Testing:**  Test the app with different configurations to ensure everything is working smoothly.  