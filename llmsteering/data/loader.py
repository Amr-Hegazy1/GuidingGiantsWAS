"""
Data loading utilities for LLM Safety Steering.

This module provides functions to load and prepare datasets for training
and evaluating the steering controller.
"""

import logging
import random
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
from datasets import load_dataset


def load_dataset_from_hf(
    dataset_name: str,
    split: str = "train",
    prompt_column: str = "prompt",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[str]:
    """Load a dataset from Hugging Face and extract prompts.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face.
        split (str): Dataset split to use (e.g., "train", "test").
        prompt_column (str): Column name containing the prompts.
        max_samples (int, optional): Maximum number of samples to load.
            If None, load all available samples.
        cache_dir (str, optional): Directory to cache the dataset.
        seed (int): Random seed for reproducibility when sampling.
        
    Returns:
        List[str]: List of prompts from the dataset.
    """
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        logging.info(f"Loaded dataset {dataset_name} (split: {split}) with {len(dataset)} examples")
        
        # Check if the prompt column exists
        if prompt_column not in dataset.column_names:
            available_columns = dataset.column_names
            logging.error(f"Column '{prompt_column}' not found in dataset. Available columns: {available_columns}")
            
            # Try to find a suitable alternative
            possible_alternatives = ["prompt", "text", "question", "instruction", "input", "query"]
            for alt in possible_alternatives:
                if alt in dataset.column_names:
                    logging.warning(f"Using '{alt}' column instead of '{prompt_column}'")
                    prompt_column = alt
                    break
            else:
                raise ValueError(f"Cannot find a suitable prompt column in dataset {dataset_name}")
        
        # Extract prompts
        prompts = [example[prompt_column] for example in dataset]
        
        # Sample if needed
        if max_samples is not None and max_samples < len(prompts):
            random.seed(seed)
            prompts = random.sample(prompts, max_samples)
            logging.info(f"Sampled {max_samples} prompts from dataset")
        
        return prompts
    
    except Exception as e:
        logging.error(f"Error loading dataset {dataset_name}: {e}")
        return []


def load_harmful_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    split: str = "train",
    prompt_column: str = "rejected",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[str]:
    """Load a dataset of harmful prompts for training.
    
    By default, uses the Anthropic/hh-rlhf dataset with rejected responses.
    
    Args:
        dataset_name (str): Name of the dataset.
        split (str): Dataset split to use.
        prompt_column (str): Column containing harmful prompts.
        max_samples (int, optional): Maximum number of samples to load.
        cache_dir (str, optional): Directory to cache the dataset.
        seed (int): Random seed for reproducibility.
        
    Returns:
        List[str]: List of harmful prompts.
    """
    return load_dataset_from_hf(
        dataset_name=dataset_name,
        split=split,
        prompt_column=prompt_column,
        max_samples=max_samples,
        cache_dir=cache_dir,
        seed=seed,
    )


def load_benign_dataset(
    dataset_name: str = "tatsu-lab/alpaca",
    split: str = "train",
    prompt_column: str = "instruction",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[str]:
    """Load a dataset of benign prompts for discriminative training.
    
    By default, uses the tatsu-lab/alpaca dataset with the instruction column,
    as specified in the project requirements.
    
    Args:
        dataset_name (str): Name of the dataset.
        split (str): Dataset split to use.
        prompt_column (str): Column containing benign prompts.
        max_samples (int, optional): Maximum number of samples to load.
        cache_dir (str, optional): Directory to cache the dataset.
        seed (int): Random seed for reproducibility.
        
    Returns:
        List[str]: List of benign prompts.
    """
    return load_dataset_from_hf(
        dataset_name=dataset_name,
        split=split,
        prompt_column=prompt_column,
        max_samples=max_samples,
        cache_dir=cache_dir,
        seed=seed,
    )


def load_and_prepare_data(
    harmful_dataset_name: str = "Anthropic/hh-rlhf",
    harmful_dataset_split: str = "train",
    harmful_prompt_column: str = "rejected",
    benign_dataset_name: Optional[str] = "tatsu-lab/alpaca",
    benign_dataset_split: Optional[str] = "train",
    benign_prompt_column: Optional[str] = "instruction",
    max_harmful_samples: Optional[int] = None,
    max_benign_samples: Optional[int] = None,
    train_ratio: float = 0.9,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """Load and prepare datasets for training and evaluation.
    
    Args:
        harmful_dataset_name (str): Name of the harmful dataset.
        harmful_dataset_split (str): Split of the harmful dataset.
        harmful_prompt_column (str): Column with harmful prompts.
        benign_dataset_name (str, optional): Name of the benign dataset.
            If None, only harmful examples will be loaded.
        benign_dataset_split (str, optional): Split of the benign dataset.
        benign_prompt_column (str, optional): Column with benign prompts.
        max_harmful_samples (int, optional): Maximum harmful samples.
        max_benign_samples (int, optional): Maximum benign samples.
        train_ratio (float): Ratio of data to use for training vs. evaluation.
        cache_dir (str, optional): Directory to cache datasets.
        seed (int): Random seed for reproducibility.
        
    Returns:
        Dict[str, Dict[str, List[str]]]: Dictionary with structure:
            {
                "train": {
                    "harmful": [...],
                    "benign": [...]
                },
                "eval": {
                    "harmful": [...],
                    "benign": [...]
                }
            }
    """
    random.seed(seed)
    
    # Load harmful dataset
    harmful_prompts = load_harmful_dataset(
        dataset_name=harmful_dataset_name,
        split=harmful_dataset_split,
        prompt_column=harmful_prompt_column,
        max_samples=max_harmful_samples,
        cache_dir=cache_dir,
        seed=seed,
    )
    
    # Split harmful prompts into train and eval
    random.shuffle(harmful_prompts)
    harmful_train_size = int(len(harmful_prompts) * train_ratio)
    harmful_train = harmful_prompts[:harmful_train_size]
    harmful_eval = harmful_prompts[harmful_train_size:]
    
    result = {
        "train": {"harmful": harmful_train},
        "eval": {"harmful": harmful_eval}
    }
    
    # Load benign dataset if specified
    if benign_dataset_name:
        benign_prompts = load_benign_dataset(
            dataset_name=benign_dataset_name,
            split=benign_dataset_split,
            prompt_column=benign_prompt_column,
            max_samples=max_benign_samples,
            cache_dir=cache_dir,
            seed=seed,
        )
        
        # Split benign prompts into train and eval
        random.shuffle(benign_prompts)
        benign_train_size = int(len(benign_prompts) * train_ratio)
        benign_train = benign_prompts[:benign_train_size]
        benign_eval = benign_prompts[benign_train_size:]
        
        result["train"]["benign"] = benign_train
        result["eval"]["benign"] = benign_eval
    
    # Log dataset sizes
    logging.info(f"Harmful train samples: {len(result['train']['harmful'])}")
    logging.info(f"Harmful eval samples: {len(result['eval']['harmful'])}")
    if benign_dataset_name:
        logging.info(f"Benign train samples: {len(result['train']['benign'])}")
        logging.info(f"Benign eval samples: {len(result['eval']['benign'])}")
    
    return result


def load_evaluation_prompts(
    dataset_names: List[str] = None,
    splits: List[str] = None,
    prompt_columns: List[str] = None,
    sample_sizes: List[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> List[str]:
    """Load and combine evaluation prompts from multiple datasets.
    
    Args:
        dataset_names (List[str], optional): Names of datasets to use.
            Default: ["Anthropic/hh-rlhf", "tatsu-lab/alpaca"]
        splits (List[str], optional): Dataset splits.
            Default: ["test", "test"]
        prompt_columns (List[str], optional): Column names for prompts.
            Default: ["rejected", "instruction"]
        sample_sizes (List[int], optional): Number of samples per dataset.
            Default: [50, 50]
        cache_dir (str, optional): Directory to cache datasets.
        seed (int): Random seed for reproducibility.
        
    Returns:
        List[str]: Combined list of evaluation prompts.
    """
    if dataset_names is None:
        dataset_names = ["Anthropic/hh-rlhf", "tatsu-lab/alpaca"]
    
    if splits is None:
        splits = ["test", "test"]
    
    if prompt_columns is None:
        prompt_columns = ["rejected", "instruction"]
    
    if sample_sizes is None:
        sample_sizes = [50, 50]
    
    # Make sure all lists have the same length
    n = len(dataset_names)
    if len(splits) != n or len(prompt_columns) != n or len(sample_sizes) != n:
        raise ValueError("All list arguments must have the same length")
    
    combined_prompts = []
    
    # Load and combine prompts from each dataset
    for i, (dataset, split, column, size) in enumerate(zip(
        dataset_names, splits, prompt_columns, sample_sizes
    )):
        logging.info(f"Loading evaluation prompts from {dataset} (split: {split}, column: {column})")
        
        prompts = load_dataset_from_hf(
            dataset_name=dataset,
            split=split,
            prompt_column=column,
            max_samples=size,
            cache_dir=cache_dir,
            seed=seed + i,  # Use different seeds for different datasets
        )
        
        combined_prompts.extend(prompts)
        logging.info(f"Added {len(prompts)} prompts from {dataset}")
    
    # Shuffle the combined prompts
    random.seed(seed)
    random.shuffle(combined_prompts)
    
    logging.info(f"Total evaluation prompts: {len(combined_prompts)}")
    
    return combined_prompts
