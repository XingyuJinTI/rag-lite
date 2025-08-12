# RAG-Minial

A minimal demonstration of Retrieval-Augmented Generation (RAG) for knowledge-based question answering using cat facts.

## Overview

This project demonstrates a simple RAG implementation that:
- Loads a dataset of cat facts from a text file
- Creates embeddings using Ollama and a BGE embedding model
- Implements semantic search using cosine similarity
- Generates responses using a language model with retrieved context

## Features

- **Simple RAG Pipeline**: Complete implementation from data loading to response generation
- **Vector Database**: In-memory vector storage with embeddings
- **Semantic Search**: Cosine similarity-based retrieval of relevant context
- **Context-Aware Responses**: Language model generates answers based on retrieved knowledge
- **Real-time Streaming**: Interactive chat interface with streaming responses

## Requirements

- Python 3.x
- [Ollama](https://ollama.ai/) installed and running locally
- Internet connection for downloading models

## Models Used

- **Embedding Model**: `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf`
- **Language Model**: `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd rag-minial
```

2. Install Ollama following the [official instructions](https://ollama.ai/download)

3. Pull the required models:
```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

4. Install Python dependencies:
```bash
pip install ollama
```

## Usage

1. Start Ollama service (if not already running):
```bash
ollama serve
```

2. Run the demo:
```bash
python demo.py
```

3. Enter your question when prompted. The system will:
   - Convert your question to an embedding
   - Find the most similar cat facts using cosine similarity
   - Retrieve the top 3 most relevant facts
   - Generate a response using the retrieved context

## How It Works

1. **Data Loading**: Reads cat facts from `cat-facts.txt`
2. **Embedding Generation**: Creates vector embeddings for each fact using the BGE model
3. **Vector Database**: Stores facts and their embeddings in memory
4. **Query Processing**: Converts user questions to embeddings
5. **Retrieval**: Finds most similar facts using cosine similarity
6. **Response Generation**: Uses retrieved context to generate relevant answers

## Project Structure

```
rag-minial/
├── demo.py          # Main RAG implementation
├── cat-facts.txt    # Dataset of cat facts
└── README.md        # This file
```

## Technical Details

- **Vector Similarity**: Cosine similarity for semantic search
- **Context Window**: Top 3 most relevant facts are used for generation
- **Streaming**: Real-time response generation for better user experience
- **Memory Efficient**: In-memory vector database suitable for small datasets

## Example

```
Ask me a question: How long do cats sleep?
Retrieved knowledge:
 - (similarity: 0.85) On average, cats spend 2/3 of every day sleeping. That means a nine-year-old cat has been awake for only three years of its life.
 - (similarity: 0.72) One reason that kittens sleep so much is because a growth hormone is released only during sleep.
 - (similarity: 0.68) Cats spend nearly 1/3 of their waking hours cleaning themselves.

Chatbot response: Based on the retrieved information, cats spend approximately 2/3 of every day sleeping, which means they sleep for about 16 hours per day...
```

## Limitations

- In-memory storage (data is lost when program exits)
- Small dataset (150 cat facts)
- Basic similarity scoring (cosine similarity only)
- No persistent vector database

## Future Improvements

- Add persistent vector database (e.g., Chroma, Pinecone)
- Implement more sophisticated retrieval methods
- Add support for different datasets
- Include evaluation metrics
- Add web interface

## License

This project is for educational and demonstration purposes.
