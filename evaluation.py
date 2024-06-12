from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric
from transformers import GenerationConfig  # For text generation
import torch

class EvaluationMetric:
    """
    Abstract base class for evaluation metrics.
    """

    def calculate(self, model_name, dataset, tokenizer=None):
        """
        Calculate and return the evaluation metric.

        Args:
            model_name (str): Name of the model to evaluate.
            dataset (datasets.Dataset): Dataset for evaluation.
            tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer to use. If not provided, it will be loaded based on the model name.

        Returns:
            dict: A dictionary containing the calculated metric value(s).
        """
        raise NotImplementedError

    def get_name(self):
        """
        Return the name of the metric.
        """
        return self.__class__.__name__


class AccuracyMetric(EvaluationMetric):
    """
    Calculates accuracy. Assumes the dataset has 'label' and 'text' fields.
    """

    def calculate(self, model_name, dataset, tokenizer=None):
        """
        Calculate accuracy.

        Args:
            model_name (str): Name of the model to evaluate.
            dataset (datasets.Dataset): Dataset for evaluation.
            tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer to use. If not provided, it will be loaded based on the model name.

        Returns:
            dict: A dictionary containing the accuracy score.
        """
        accuracy_metric = load_metric("accuracy", keep_in_memory=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # Generate predictions from the model
        with torch.no_grad():
            logits = model(**tokenized_dataset).logits
        predictions = torch.argmax(logits, dim=-1)
        tokenized_dataset["predictions"] = predictions

        # Calculate accuracy
        results = accuracy_metric.compute(
            predictions=tokenized_dataset["predictions"],
            references=tokenized_dataset["label"],
        )
        return results


class PerplexityMetric(EvaluationMetric):
    """
    Calculates perplexity. Assumes the dataset has a 'text' field.
    """

    def calculate(self, model_name, dataset, tokenizer=None):
        """
        Calculate perplexity.

        Args:
            model_name (str): Name of the model to evaluate.
            dataset (datasets.Dataset): Dataset for evaluation.
            tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer to use. If not provided, it will be loaded based on the model name.

        Returns:
            dict: A dictionary containing the perplexity score.
        """
        perplexity_metric = load_metric("perplexity", keep_in_memory=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # Generate logits from the model
        with torch.no_grad():
            logits = model(**tokenized_dataset).logits
        tokenized_dataset["logits"] = logits

        # Calculate perplexity
        results = perplexity_metric.compute(
            predictions=tokenized_dataset["logits"],
            references=tokenized_dataset["input_ids"],
        )
        return results


class RougeMetric(EvaluationMetric):
    """
    Calculates ROUGE scores. Assumes the dataset has 'prompt' and 'completion' fields.
    """

    def calculate(self, model_name, dataset, tokenizer=None):
        """
        Calculate ROUGE.

        Args:
            model_name (str): Name of the model to evaluate.
            dataset (datasets.Dataset): Dataset for evaluation.
            tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer to use. If not provided, it will be loaded based on the model name.

        Returns:
            dict: A dictionary containing the ROUGE scores.
        """
        rouge_metric = load_metric("rouge", keep_in_memory=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Generation configuration - adjust these if needed
        generation_config = GenerationConfig(
            max_new_tokens=64,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )

        def generate_summary(examples):
            # Tokenize the prompts
            inputs = tokenizer(
                examples["prompt"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Generate summaries using the model
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=generation_config)

            # Decode the generated summaries
            examples[
                "completion"
            ] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return examples

        dataset = dataset.map(generate_summary, batched=True)
        results = rouge_metric.compute(
            predictions=dataset["completion"],
            references=dataset["prompt"],
            use_stemmer=True,
        )
        # Extract the mid values and format to percentages
        results = {key: value.mid.fmeasure * 100 for key, value in results.items()}
        return results


def get_evaluation_metric(metric_name, config):
    """
    Return the evaluation metric class based on the name.

    Args:
        metric_name (str): Name of the evaluation metric.
        config (dict): The configuration dictionary.

    Returns:
        EvaluationMetric: The evaluation metric class.
    """
    metric_class = config["evaluation_metrics"][metric_name]["class"]
    return globals()[metric_class]()