from setuptools import setup, find_packages

setup(
    name="llmsteering",
    version="0.1.0",
    description="A framework for steering Large Language Models to provide safe responses",
    author="LLM Safety Steering Team",
    author_email="author@example.com",
    url="https://github.com/yourusername/llm-safety-steering",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.25.0",
        "accelerate>=0.17.0",
        "datasets>=2.7.0",
        "pyyaml>=6.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "einops>=0.4.1",
        "sentencepiece>=0.1.97",
        "openai>=1.0.0",  # For refusal rate calculation
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
