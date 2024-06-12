from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSeq2SeqLM,
)
from trl import (
    create_reference_model,
    AutoModelForCausalLMWithValueHead,
    SFTConfig,
    DPOConfig,
    PPOConfig,
    LoraConfig, 
    get_peft_model,
)
from utils import format_metrics_output
from evaluation import RougeMetric, AccuracyMetric, PerplexityMetric
import torch

# Load configuration (Needed to access paths)
with open("config.json", "r") as f:
    CONFIG = json.load(f)

class FineTuningMethod:
    """
    Abstract base class for fine-tuning methods.
    """

    def train(self, model_name, dataset, data_collator, learning_rate, epochs, **kwargs):
        """
        Fine-tune the model using the specified method and parameters.

        Args:
            model_name (str): Name of the model to fine-tune.
            dataset (datasets.Dataset): Dataset for fine-tuning.
            data_collator (DataCollator): The data collator to use.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of epochs to train.
            **kwargs: Additional keyword arguments for the specific method.

        Returns:
            tuple: A tuple containing the fine-tuned model and a dictionary of evaluation metrics.
        """
        raise NotImplementedError

    def get_metrics(self, model, dataset, tokenizer):
        """
        Evaluate the model and return a dictionary of metrics.

        Args:
            model: The fine-tuned model.
            dataset (datasets.Dataset): Dataset for evaluation.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for the model.

        Returns:
            dict: A dictionary of evaluation metrics.
        """
        raise NotImplementedError


class SFTMethod(FineTuningMethod):
    """
    Fine-tuning method using Supervised Fine-Tuning (SFT).
    """

    def train(self, model_name, dataset, data_collator, learning_rate, epochs, lora_r=None, lora_alpha=None, lora_dropout=None, **kwargs):
        """
        Fine-tune using Supervised Fine-Tuning (SFT).
        """
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Get lora_r, lora_alpha, lora_dropout from kwargs or config
        if lora_r is None:
            lora_r = CONFIG["fine_tuning_methods"]["SFT"]["lora_r"]
        if lora_alpha is None:
            lora_alpha = CONFIG["fine_tuning_methods"]["SFT"]["lora_alpha"]
        if lora_dropout is None:
            lora_dropout = CONFIG["fine_tuning_methods"]["SFT"]["lora_dropout"]

        # Apply LoRA
        if lora_r > 0:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=CONFIG["fine_tuning_methods"]["SFT"]["output_dir"],
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 8),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            # ... add other training arguments from kwargs ...
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=dataset,  # Use same data for eval for now
        )
        trainer.train()

        return trainer.model, self.get_metrics(trainer.model, dataset, tokenizer)

    def get_metrics(self, model, dataset, tokenizer):
        """
        Get metrics for SFT. Currently only returns training loss.
        """
        # For simplicity, we'll just return training loss for now
        metrics = {"training_loss": model.trainer.state.log_history[-1]["train_loss"]}

        # Add ROUGE metric if it's a summarization task
        task_type = CONFIG["datasets"][kwargs["dataset_name"]]["preparation"]["type"]
        if task_type == "summarization":
            rouge_metric = RougeMetric()
            rouge_results = rouge_metric.calculate(model, dataset, tokenizer)
            metrics.update(rouge_results)

        return metrics


class DPOMethod(FineTuningMethod):
    """
    Fine-tuning method using Direct Preference Optimization (DPO).
    """

    def train(
        self,
        model_name,
        dataset,
        data_collator,
        learning_rate,
        epochs,
        beta=None,
        lora_r=None,
        lora_alpha=None,
        lora_dropout=None,
        **kwargs
    ):
        """
        Fine-tune using Direct Preference Optimization (DPO).
        """
        model = AutoModelForCausalLM.from_pretrained(model_name)
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Get beta, lora_r, lora_alpha, lora_dropout from kwargs or config
        if beta is None:
            beta = CONFIG["fine_tuning_methods"]["DPO"]["default_beta"]
        if lora_r is None:
            lora_r = CONFIG["fine_tuning_methods"]["DPO"]["lora_r"]
        if lora_alpha is None:
            lora_alpha = CONFIG["fine_tuning_methods"]["DPO"]["lora_alpha"]
        if lora_dropout is None:
            lora_dropout = CONFIG["fine_tuning_methods"]["DPO"]["lora_dropout"]

        # Apply LoRA
        if lora_r > 0:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            ref_model = get_peft_model(
                ref_model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=CONFIG["fine_tuning_methods"]["DPO"]["output_dir"],
            max_steps=epochs * len(dataset),
            learning_rate=learning_rate,
            beta=beta,
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 8),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            # ... add other training arguments from kwargs ...
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        trainer.train()

        return trainer.model, self.get_metrics(trainer.model, dataset, tokenizer)

    def get_metrics(self, model, dataset, tokenizer):
        """
        Get metrics for DPO. Calculates the average reward for chosen and rejected responses.

        Args:
            model: The fine-tuned model.
            dataset (datasets.Dataset): Dataset for evaluation.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for the model.

        Returns:
            dict: A dictionary of evaluation metrics.
        """
        # Get the reward model name for DPO from config
        reward_model_name = CONFIG["fine_tuning_methods"]["DPO"].get(
            "reward_model_name"
        )
        if reward_model_name is None:
            raise ValueError(
                "Reward model name for DPO is not defined in config.json."
            )

        # Load the reward model and tokenizer
        reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            reward_model_name
        )
        tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

        # Tokenize the dataset
        def preprocess_function(examples):
            return tokenizer(
                examples["prompt"], padding="max_length", truncation=True
            )

        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # Calculate rewards
        rewards_chosen = []
        rewards_rejected = []
        for batch in tokenized_dataset:
            with torch.no_grad():
                outputs_chosen = reward_model(
                    **batch,
                    input_ids=tokenizer(
                        batch["chosen"], padding="max_length", truncation=True
                    ).input_ids,
                )
                rewards_chosen.append(outputs_chosen.rewards.mean().item())
                outputs_rejected = reward_model(
                    **batch,
                    input_ids=tokenizer(
                        batch["rejected"], padding="max_length", truncation=True
                    ).input_ids,
                )
                rewards_rejected.append(outputs_rejected.rewards.mean().item())

        # Also add ROUGE metric if it's a summarization task
        metrics = {
            "mean_reward_chosen": sum(rewards_chosen) / len(rewards_chosen),
            "mean_reward_rejected": sum(rewards_rejected) / len(rewards_rejected),
        }
        task_type = CONFIG["datasets"][kwargs["dataset_name"]]["preparation"]["type"]
        if task_type == "summarization":
            rouge_metric = RougeMetric()
            rouge_results = rouge_metric.calculate(model, dataset, tokenizer)
            metrics.update(rouge_results)

        return metrics

class PPOMethod(FineTuningMethod):
    """
    Fine-tuning method using Proximal Policy Optimization (PPO).
    """

    def train(
        self,
        model_name,
        dataset,
        data_collator,
        learning_rate,
        epochs,
        reward_model_name,
        lora_r=None,
        lora_alpha=None,
        lora_dropout=None,
        **kwargs
    ):
        """
        Fine-tune using Proximal Policy Optimization (PPO).
        """
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        ref_model = create_reference_model(model)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set padding token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Get lora_r, lora_alpha, lora_dropout from kwargs or config
        if lora_r is None:
            lora_r = CONFIG["fine_tuning_methods"]["PPO"]["lora_r"]
        if lora_alpha is None:
            lora_alpha = CONFIG["fine_tuning_methods"]["PPO"]["lora_alpha"]
        if lora_dropout is None:
            lora_dropout = CONFIG["fine_tuning_methods"]["PPO"]["lora_dropout"]

        # Apply LoRA
        if lora_r > 0:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            ref_model = get_peft_model(
                ref_model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=CONFIG["fine_tuning_methods"]["PPO"]["output_dir"],
            total_ppo_epochs=epochs,
            learning_rate=learning_rate,
            reward_model=reward_model_name,
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 8),
            gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            # ... add other training arguments from kwargs ...
        )

        trainer = PPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            dataset=dataset,
        )
        trainer.train()

        return trainer.model, self.get_metrics(trainer.model, dataset, tokenizer)

    def get_metrics(self, model, dataset, tokenizer):
        """
        Get metrics for PPO. Calculates the average reward and KL divergence.

        Args:
            model: The fine-tuned model.
            dataset (datasets.Dataset): Dataset for evaluation.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for the model.

        Returns:
            dict: A dictionary of evaluation metrics.
        """
        # Load the reference model
        ref_model = create_reference_model(model)

        # Tokenize the dataset
        def preprocess_function(examples):
            return tokenizer(
                examples["prompt"], padding="max_length", truncation=True
            )

        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # Calculate rewards and KL divergence
        rewards = []
        kl_divs = []
        for batch in tokenized_dataset:
            with torch.no_grad():
                outputs = model(**batch)
                ref_outputs = ref_model(**batch)
                rewards.append(outputs.rewards.mean().item())
                kl_divs.append(
                    outputs.log_probs.mean().item()
                    - ref_outputs.log_probs.mean().item()
                )

        # Also add ROUGE metric if it's a summarization task
        metrics = {
            "mean_reward": sum(rewards) / len(rewards),
            "kl_divergence": sum(kl_divs) / len(kl_divs),
        }
        task_type = CONFIG["datasets"][kwargs["dataset_name"]]["preparation"]["type"]
        if task_type == "summarization":
            rouge_metric = RougeMetric()
            rouge_results = rouge_metric.calculate(model, dataset, tokenizer)
            metrics.update(rouge_results)

        return metrics

def get_fine_tuning_method(method_name, config):
    """
    Return the fine-tuning method class based on the name.
    """
    method_class = config["fine_tuning_methods"][method_name]["class"]
    return globals()[method_class]()