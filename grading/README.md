# LLM Quiz Challenge Library

A Python library and CLI tool for creating educational quiz challenges where students try to stump AI models. Students create challenging questions, and the system evaluates whether they can outsmart Large Language Models (LLMs).

## 🎯 Overview

The LLM Quiz Challenge system works by:
1. **Students create quiz questions** in TOML format with correct answers
2. **Context loading** from URLs provides relevant course materials to the LLM
3. **Question validation** ensures appropriateness and quality
4. **Quiz-taking LLM** attempts to answer questions using the provided context
5. **Evaluator LLM** judges whether the quiz-taker's response is correct
6. **Students win** if they stump the AI on their questions (100% win rate required)
7. **Automatic grading** provides pass/fail results for GitHub Classroom integration

## 🚀 Quick Start

### Basic Usage with uv

```bash
# From grading-toolkit directory
cd grading-toolkit

# Run with specified models, context URLs, and API key
uv run python -m llm_quiz \
  --quiz-file quiz.toml \
  --api-key sk-or-v1-your-openrouter-key \
  --quiz-model gpt-4o-mini \
  --evaluator-model gpt-4o \
  --context-urls context_urls.txt

# Or use a configuration file (recommended)
uv run python -m llm_quiz \
  --quiz-file quiz.toml \
  --api-key sk-or-v1-your-openrouter-key \
  --config config.toml
```

### Required Parameters

- `--quiz-file`: Path to TOML file containing your questions
- `--api-key`: Your OpenRouter API key (or other LLM provider)
- `--quiz-model`: Model for answering questions (e.g., `gpt-4o-mini`)
- `--evaluator-model`: Model for evaluating answers (e.g., `gpt-4o`)
- `--context-urls`: File containing URLs with course materials (one URL per line)

### Optional Parameters

- `--config`: Path to TOML configuration file (recommended)
- `--context-window-size`: Context window size for LLM models (default: 32768)
- `--max-tokens`: Maximum tokens in LLM responses (default: 500)
- `--output`: Save detailed results to JSON file
- `--verbose`: Enable detailed logging

### Example Commands

```bash
# Using configuration file (recommended)
uv run python -m llm_quiz \
  --quiz-file my_quiz.toml \
  --api-key sk-or-v1-abc123 \
  --config config.toml

# Basic usage without config file
uv run python -m llm_quiz \
  --quiz-file my_quiz.toml \
  --api-key sk-or-v1-abc123 \
  --quiz-model gpt-4o-mini \
  --evaluator-model gpt-4o \
  --context-urls lecture_urls.txt

# Override config file values with command line options
uv run python -m llm_quiz \
  --quiz-file my_quiz.toml \
  --api-key sk-or-v1-abc123 \
  --config config.toml \
  --max-tokens 1000 \
  --verbose

# Save detailed results
uv run python -m llm_quiz \
  --quiz-file my_quiz.toml \
  --api-key sk-or-v1-abc123 \
  --config config.toml \
  --output results.json
```

## 📁 File Setup

### 1. Configuration File (TOML format - Recommended)
Create a `config.toml` file to organize your parameters and context URLs:

```toml
[api]
base_url = "https://openrouter.ai/api/v1"

[models]
quiz_model = "gpt-4o-mini"
evaluator_model = "gpt-4o"          # Must support structured outputs

[parameters]
context_window_size = 32768
max_tokens = 750                    # Increased from default 500

[context]
urls = [
    "https://raw.githubusercontent.com/user/repo/main/lecture-01.md",
    "https://raw.githubusercontent.com/user/repo/main/lecture-02.md",
    "https://docs.example.com/course-materials.html"
]

[output]
results_file = "quiz_results.json"
verbose = false
```

**Benefits of using config files:**
- ✅ Organize all parameters in one place
- ✅ Reusable configurations for different courses/modules
- ✅ No need to remember long command lines
- ✅ Easy to share configurations with others
- ✅ Command line arguments still override config values

### 2. Quiz File (TOML format)
Create a `quiz.toml` file with your questions:

```toml
title = "My Quiz"

[[questions]]
question = "Your challenging question here?"
answer = "Complete and accurate answer here."

[[questions]]
question = "Another challenging question?"
answer = "Another complete answer."
```

### 3. Context URLs File (Alternative to config file)
If not using a config file, create a `context_urls.txt` file with course material URLs:

```txt
# Course materials for context (one URL per line)
https://raw.githubusercontent.com/user/repo/main/lecture-01.md
https://raw.githubusercontent.com/user/repo/main/lecture-02.md
https://docs.example.com/course-materials.html
```

### 4. API Key
Get your API key from [OpenRouter](https://openrouter.ai/) or your preferred LLM provider.

## 🎯 Creating Effective Questions

### Question Types That Stump LLMs
- **Edge cases**: Boundary conditions, exceptions, unusual scenarios
- **Subtle distinctions**: Fine differences between similar concepts
- **Multi-constraint problems**: Questions combining multiple factors
- **Counterintuitive examples**: Results that seem wrong but are correct
- **Context-specific details**: Questions requiring knowledge of provided materials

### Question Types LLMs Handle Easily
- Basic definitions and explanations
- Standard textbook examples
- General conceptual overviews
- Simple step-by-step procedures

## ✅ Question Validation

Questions are automatically validated for:
- ❌ **Heavy math** (no complex derivations)
- ❌ **Prompt injection** (no AI manipulation attempts)
- ✅ **Answer quality** (accurate and complete)
- ✅ **Context relevance** (when context materials are provided)

## 🔧 Advanced Features

### Structured Outputs
The system uses OpenAI-compatible structured outputs for reliable evaluation and validation:
- **JSON Schema validation** ensures consistent responses
- **Type-safe parsing** eliminates parsing errors
- **Compatible models**: OpenAI GPT-4o+, Fireworks models, and other structured output providers

### Configurable Parameters
- **Max tokens**: Control response length (default: 500, configurable via `--max-tokens` or config file)
- **Context window size**: Adjust for different models (default: 32768)
- **Multiple model support**: Use different models for quiz-taking and evaluation

### Configuration Management
- **TOML configuration files** for organized parameter management
- **Command line override** of config file values
- **Environment-specific configs** (development, production, different courses)
- **Example configurations** included in `config_examples/` directory

## 🎓 GitHub Classroom Integration

### Automatic Grading
- **Exit Code 0**: Student passes (100% win rate on valid questions)
- **Exit Code 1**: Student fails (less than 100% win rate or validation issues)

### Grade Markers
- `STUDENTS_QUIZ_KEIKO_WIN`: Student passed
- `STUDENTS_QUIZ_KEIKO_LOSE`: Student failed

## 📋 Requirements

- Python 3.7+
- [uv](https://docs.astral.sh/uv/) package manager
- API access to OpenRouter or compatible LLM provider

### Installing uv
```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip
pip install uv
```

---

**Happy Quiz Creation!** 🎯🤖