import os
import json
import logging
import gradio as gr
from utils import (
    get_model_summary,
    install_flash_attn,
    get_available_options,
    format_metrics_output,
)
from fine_tuning import FineTuningMethod, get_fine_tuning_method
from evaluation import EvaluationMetric, get_evaluation_metric
from datasets import load_and_prepare_dataset, tokenize_and_collate
from joblib import Memory

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Install flash attention
install_flash_attn()

# Load configuration
with open("config.json", "r") as f:
    CONFIG = json.load(f)

# Available options from config
DATASETS = get_available_options("datasets", CONFIG)
TRL_METHODS = get_available_options("fine_tuning_methods", CONFIG)
EVALUATION_METRICS = get_available_options("evaluation_metrics", CONFIG)
TOKENIZATION_STRATEGIES = get_available_options("tokenization_strategies", CONFIG)
COLLATION_STRATEGIES = get_available_options("collation_strategies", CONFIG)

# Default training arguments
DEFAULT_TRAINING_ARGS = {
    "output_dir": "finetuned_model",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 5e-5,
    "evaluation_strategy": "epoch",
    # ... add more default arguments as needed ...
}

# Gradio interface
with gr.Blocks(theme="sudeepshouche/minimalist") as demo:
    with gr.Row():
        with gr.Column():
            model_name = gr.Textbox(
                label="Model Name",
                placeholder="Enter model name or select example...",
                lines=1,
            )

            for category, examples in CONFIG["model_examples"].items():
                gr.Markdown(f"### {category}")
                gr.Examples(examples=examples, inputs=model_name)

            submit_button = gr.Button("Submit")

            # Fine-tuning Section
            with gr.Accordion("Fine-tuning Options", open=False):
                # Dataset Input (CSV upload)
                dataset_input = gr.File(label="Upload Dataset (CSV)")
                
                # or select from Hugging Face Hub
                dataset_dropdown = gr.Dropdown(choices=DATASETS, label="Select from Hub", value="None") #  Set default value to None

                method_radio = gr.Radio(
                    choices=TRL_METHODS,
                    label="Fine-tuning Method",
                    info="Select the desired fine-tuning method.",
                )

                # Tokenization and Collation
                tokenization_dropdown = gr.Dropdown(
                    choices=TOKENIZATION_STRATEGIES,
                    label="Tokenization Strategy",
                    info="Choose the tokenization strategy",
                )
                collation_dropdown = gr.Dropdown(
                    choices=COLLATION_STRATEGIES,
                    label="Collation Strategy",
                    info="Choose the collation strategy",
                )

                # Training Arguments
                training_args = {}
                for arg, default_value in DEFAULT_TRAINING_ARGS.items():
                    component = gr.Number(
                        label=arg.replace("_", " ").title(),
                        value=default_value,
                        info=f"Specify the {arg} value.",
                    )
                    training_args[arg] = component

                # Add hyperparameter components dynamically based on config
                hyperparam_components = {}
                for method in TRL_METHODS:
                    with gr.Tab(method):
                        for param, default_value in CONFIG["fine_tuning_methods"][
                            method
                        ].items():
                            if param not in [
                                "class",
                                "trainer",
                                "config",
                                "output_dir",
                                "learning_rate",
                                "epochs",
                            ]:
                                component = gr.Number(
                                    label=param.replace("_", " ").title(),
                                    value=default_value,
                                    info=f"Specify the {param} value.",
                                )
                                hyperparam_components[(method, param)] = component

            fine_tune_button = gr.Button("Fine-tune Model")

        with gr.Column():
            model_summary = gr.Textbox(
                label="Model Architecture",
                lines=20,
                placeholder="Model architecture will appear here...",
                show_copy_button=True,
            )
            error_output = gr.Textbox(
                label="Error",
                lines=10,
                placeholder="Exceptions will appear here...",
                show_copy_button=True,
            )

            # Evaluation Section
            with gr.Accordion("Evaluation Results", open=False):
                metric_dropdown = gr.Dropdown(
                    choices=EVALUATION_METRICS, label="Evaluation Metric"
                )
                evaluation_metrics = gr.Textbox(
                    label="Metrics",
                    lines=5,
                    placeholder="Evaluation metrics will appear here...",
                )

    def handle_submit(model_name):
        logging.info(f"Loading model: {model_name}")
        summary, error = get_model_summary(model_name)
        return summary, error

    def handle_fine_tune(
        model_name,
        dataset_file,
        dataset_hub,
        method,
        tokenization,
        collation,
        lora_r,
        lora_alpha,
        lora_dropout,
        reward_model_name,
        num_train_epochs,
        per_device_train_batch_size,
        gradient_accumulation_steps,
        **training_kwargs
    ):
        try:
            if dataset_file is not None:
                dataset = load_and_prepare_dataset(
                    dataset_file.name, CONFIG, source="file"
                )
                logging.info(
                    f"Fine-tuning {model_name} on uploaded dataset using {method}"
                )
            elif dataset_hub != "None":
                dataset = load_and_prepare_dataset(
                    dataset_hub, CONFIG, source="hub"
                )
                logging.info(
                    f"Fine-tuning {model_name} on {dataset_hub} dataset using {method}"
                )
            else:
                raise ValueError(
                    "Please upload a dataset or select one from the Hub."
                )

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

            # Combine hyperparameters and training arguments
            training_kwargs.update(training_args_dict)

            # Tokenize and collate the dataset
            tokenized_dataset, data_collator = tokenize_and_collate(
                dataset,
                model_name,
                tokenization,
                collation,
                CONFIG,
                dataset_name=dataset_hub if dataset_hub != "None" else dataset_file.name,
                **training_kwargs
            )

            ft_method = get_fine_tuning_method(method, CONFIG)
            trained_model, metrics = ft_method.train(
                model_name,
                tokenized_dataset,
                data_collator,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                reward_model_name=reward_model_name,
                **training_kwargs
            )

            # Save the fine-tuned model and tokenizer
            trained_model.save_pretrained(
                CONFIG["fine_tuning_methods"][method]["output_dir"]
            )
            tokenizer.save_pretrained(
                CONFIG["fine_tuning_methods"][method]["output_dir"]
            )

            return str(trained_model), format_metrics_output(metrics)
        except Exception as e:
            logging.error(f"Error during fine-tuning: {e}")
            return "", str(e)

    def handle_evaluate(model_name, metric):
        try:
            logging.info(f"Evaluating {model_name} using {metric}")
            eval_metric = get_evaluation_metric(metric, CONFIG)
            dataset = load_and_prepare_dataset(
                dataset_dropdown.value, CONFIG, source="hub"
            )
            metrics = eval_metric.calculate(model_name, dataset)
            return format_metrics_output(metrics)
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            return str(e)

    submit_button.click(
        fn=handle_submit, inputs=model_name, outputs=[model_summary, error_output]
    )
    fine_tune_button.click(
        fn=handle_fine_tune,
        inputs=[
            model_name,
            dataset_input,
            dataset_dropdown,
            method_radio,
            tokenization_dropdown,
            collation_dropdown,
            lora_r,
            lora_alpha,
            lora_dropout,
            reward_model_name,
            num_train_epochs,
            per_device_train_batch_size,
            gradient_accumulation_steps,
        ]
        + list(training_args.values())
        + list(hyperparam_components.values()),
        outputs=[model_summary, evaluation_metrics],
    )
    metric_dropdown.change(
        fn=handle_evaluate,
        inputs=[model_name, metric_dropdown],
        outputs=[evaluation_metrics],
    )

# Launch the interface
demo.launch()