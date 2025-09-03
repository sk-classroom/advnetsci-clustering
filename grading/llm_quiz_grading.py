#!/usr/bin/env python3
"""
LLM Quiz Challenge Script

This script is a simple wrapper around the llm_quiz library that implements
a quiz challenge where students try to stump AI models.

Usage:
    python llm_quiz_grading.py --quiz-file quiz.toml --base-url https://api.example.com --api-key your-api-key

Required Arguments:
    --base-url: LLM API base URL
    --api-key: API key for the LLM endpoint

This script now uses the refactored llm_quiz library for all core functionality.
For direct library usage:
    from llm_quiz import LLMQuizChallenge
"""

# Import the CLI module from the refactored library
from llm_quiz.cli import main

if __name__ == "__main__":
    main()