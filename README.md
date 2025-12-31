# 🔍 FAISS Embedder - Vector Search for Your Documents

> **Fast semantic search for your documents using FAISS + Ollama embeddings**

[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-79%20passing-brightgreen.svg)](#testing)
[![npm](https://img.shields.io/npm/v/faiss-embedder.svg)](https://www.npmjs.com/package/faiss-embedder)

A CLI tool and library for building and querying FAISS vector indexes from your documents. Uses **Ollama** for local embeddings (no API costs!) and **FAISS** for blazing-fast similarity search.

---

## ✨ Features

- 🚀 **Fast** - FAISS-powered similarity search in milliseconds
- 💰 **Free** - Uses Ollama for local embeddings (no API costs)
- 🔧 **Simple CLI** - Build and query indexes with simple commands
- 📦 **Programmatic API** - Use as a library in your Node.js projects
- 🔄 **Change Detection** - MD5 hashing to detect document changes
- 📊 **Multiple Models** - Support for various Ollama embedding models
- 📝 **Smart Splitting** - LangChain text splitters for Markdown, code, and more

---

## 📋 Prerequisites

- **Node.js** 18+
- **Ollama** installed and running

\`\`\`bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama
ollama serve

# Pull embedding model
ollama pull nomic-embed-text
\`\`\`

---

## 🚀 Installation

### npm (recommended)

\`\`\`bash
npm install faiss-embedder
\`\`\`

### Global CLI

\`\`\`bash
npm install -g faiss-embedder

# Then use anywhere
faiss-gen build ./docs
faiss-gen query ./output "search query"
\`\`\`

### From source

\`\`\`bash
git clone https://github.com/abuzarhamza/faiss-embedder
cd faiss-embedder
npm install
\`\`\`

---

## ⚡ Quick Start

### 1. Build an index from documents

\`\`\`bash
faiss-gen build ./documents ./my_index
\`\`\`

### 2. Query the index

\`\`\`bash
faiss-gen query ./my_index "find orders by status"
\`\`\`

### 3. View configuration

\`\`\`bash
faiss-gen config
\`\`\`

---

## 🖥️ CLI Usage

### Commands

| Command | Description |
|---------|-------------|
| \`build <input-dir> [output-dir]\` | Build FAISS index from documents |
| \`query <index-dir> <query>\` | Search the index |
| \`config\` | Show settings and available models |

### Build Command

\`\`\`bash
faiss-gen build <input-dir> [output-dir] [options]
\`\`\`

**Options:**

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| \`--chunk-size\` | \`-c\` | 1500 | Chunk size in characters |
| \`--overlap\` | \`-o\` | 200 | Overlap between chunks |
| \`--extensions\` | \`-e\` | \`.txt,.md,.js,.json\` | File extensions to include |
| \`--recursive\` | \`-r\` | false | Scan subdirectories |
| \`--index-type\` | \`-t\` | IP | Index type (IP/L2) |
| \`--model\` | \`-m\` | nomic-embed-text | Embedding model |
| \`--splitter\` | \`-s\` | recursive | Text splitter type |
| \`--ollama-url\` | | localhost:11434 | Ollama server URL |

**Examples:**

\`\`\`bash
# Basic usage
faiss-gen build ./docs

# Custom chunk size and extensions
faiss-gen build ./src ./code_index -c 1000 -e .js,.ts,.py

# Recursive with different model
faiss-gen build ./data -r -m mxbai-embed-large
\`\`\`

### Query Command

\`\`\`bash
faiss-gen query <index-dir> <query> [options]
\`\`\`

**Options:**

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| \`--top-k\` | \`-k\` | 5 | Number of results |
| \`--show-chunk\` | | true | Show chunk content |
| \`--max-length\` | | 500 | Max chars per chunk |

### Config Command

\`\`\`bash
# Show configuration
faiss-gen config

# Check Ollama status
faiss-gen config --check
\`\`\`

---

## 📦 Programmatic API

\`\`\`javascript
import { query, build, FaissIndexer } from 'faiss-embedder';

// Query an existing index
const results = await query('./my_index', 'find orders by status', {
  topK: 5,
  model: 'nomic-embed-text'
});

results.forEach(r => {
  console.log(\`Score: \${r.score.toFixed(4)}\`);
  console.log(\`Doc: \${r.doc}\`);
  console.log(\`Content: \${r.chunk.substring(0, 100)}...\`);
});

// Build an index
const result = await build('./documents', './output', {
  chunkSize: 1000,
  overlap: 200,
  extensions: ['.md', '.txt'],
  recursive: true
});

console.log(\`Built \${result.vectors} vectors in \${result.time}ms\`);
\`\`\`

---

## 📝 Text Splitters

Uses \`@langchain/textsplitters\` for intelligent text chunking.

| Type | Description | Best For |
|------|-------------|----------|
| **recursive** ⭐ | Respects paragraphs/sentences | General text (default) |
| **markdown** | Respects headers, code blocks | \`.md\` files |
| **code** | Respects functions, classes | Source code |

---

## 🔧 Available Embedding Models

| Model | Dimension | Description |
|-------|-----------|-------------|
| **nomic-embed-text** ⭐ | 768 | Default, fast, general-purpose |
| **mxbai-embed-large** | 1024 | Higher quality |
| **all-minilm** | 384 | Lightweight, fastest |

---

## 📁 Output Files

\`\`\`
output_dir/
├── index.bin              # FAISS binary index
├── index_metadata.json    # Chunk metadata
├── doc_index_cache.json   # MD5 hashes for change detection
└── metadata.json          # Raw chunked data
\`\`\`

---

## 🧪 Testing

\`\`\`bash
npm test
\`\`\`

79 tests passing (docCache, embedder, faissIndexer, textSplitter)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [Ollama](https://ollama.ai) - Local LLM runner
- [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1) - Embedding model
- [LangChain](https://js.langchain.com/) - Text splitters
