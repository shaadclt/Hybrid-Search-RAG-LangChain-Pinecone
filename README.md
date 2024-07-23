# Hybrid Search RAG using LangChain and Pinecone in Google Colab

This repository contains a Google Colab notebook that demonstrates how to set up and use a hybrid search Retrieval-Augmented Generation (RAG) system using LangChain and Pinecone. The hybrid search combines vector embeddings and sparse (BM25) encodings to provide efficient and accurate information retrieval.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project showcases how to create a hybrid search system that leverages both vector embeddings and sparse encodings for enhanced information retrieval. We use LangChain for managing the embeddings and Pinecone for creating and managing the vector index. The system can efficiently retrieve relevant information from a set of sentences based on a given query.

## Prerequisites

To run this project, you need the following:
- Google Colab account
- Pinecone API key
- Basic knowledge of Python

## Setup

Follow these steps to set up and run the project:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/shaadclt/Hybrid-Search-RAG-LangChain-Pinecone.git
   cd Hybrid-Search-RAG-LangChain-Pinecone
   ```

2. **Open Google Colab:**

Upload the provided notebook (hybrid_search_RAG.ipynb) to your Google Colab account.

3. **Install Dependencies:**

Run the following commands in the Colab notebook to install the required libraries:

```bash
!pip install --upgrade --quiet pinecone-client pinecone-text pinecone-notebooks
!pip install langchain-community -q
!pip install langchain-huggingface -q
```

4. **Initialize Pinecone:**

Retrieve your Pinecone API key from Google Colab's user data and initialize the Pinecone client:
```bash
from google.colab import userdata
api_key = userdata.get('PINECONE_API')
from pinecone import Pinecone, ServerlessSpec

index_name = "hybrid-search-langchain-pinecone"
pc = Pinecone(api_key=api_key)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=384, 
        metric="dotproduct",
        spec=ServerlessSpec(cloud='aws', region="us-east-1")
    )
index = pc.Index(index_name)
```

5. **Vector Embedding and Sparse Encoding:**

Set up the vector embeddings and sparse encoding:

```bash
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()

sentences = [
    "In 2023, I visited Paris",
    "In 2022, I visited New York",
    "In 2021, I visited London",
    "In 2020, I visited Rome"
]

bm25_encoder.fit(sentences)
bm25_encoder.dump("bm25_values.json")
bm25_encoder = BM25Encoder().load("bm25_values.json")
```

6. **Create the Retriever and Add Texts:**

Initialize the retriever and add texts to the index:

```bash
from langchain_community.retrievers import PineconeHybridSearchRetriever

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
retriever.add_texts(sentences)
```

7. **Invoke the Retriever:**

Use the retriever to query the index:

```bash
result = retriever.invoke("What city did I visit in 2022?")[0]
print(result)
```

## Usage
After setting up the project, you can run the provided Colab notebook to see how the hybrid search system works. The notebook demonstrates the following steps:

- Initializing the Pinecone client
- Creating and managing the Pinecone index
- Setting up vector embeddings and sparse encodings
- Adding texts to the Pinecone index
- Querying the index using the hybrid search retriever

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
   
