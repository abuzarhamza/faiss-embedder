/**
 * FAISS Index Generator - Public API
 *
 * Usage:
 *   import { query, build, FaissIndexer } from 'faiss-index-gen';
 *
 *   // Query an existing index
 *   const results = await query('./faiss_output', 'find orders by status');
 *
 *   // Build index programmatically
 *   await build('./documents', './output');
 */

// Core exports
export { FaissIndexer, buildIndex } from "./lib/faissIndexer.js";
export { OllamaEmbedder } from "./lib/embedder.js";
export {
	generateDocCache,
	loadDocCache,
	detectChanges,
	getFileHash,
} from "./lib/docCache.js";
export {
	splitText,
	splitTextByFileType,
	chunkText,
	createSplitter,
	SPLITTER_TYPES,
} from "./lib/textSplitter.js";

import { FaissIndexer } from "./lib/faissIndexer.js";
import { splitTextByFileType } from "./lib/textSplitter.js";
import fs from "fs";
import path from "path";

/**
 * Query an existing FAISS index
 *
 * @param {string} indexDir - Directory containing index.bin and index_metadata.json
 * @param {string} queryText - Search query
 * @param {Object} [options]
 * @param {number} [options.topK=5] - Number of results
 * @param {string} [options.model='nomic-embed-text'] - Embedding model
 * @param {string} [options.baseUrl='http://localhost:11434'] - Ollama URL
 * @returns {Promise<Array<{id, score, doc, chunk_id, chunk}>>}
 *
 * @example
 * const results = await query('./faiss_output', 'find orders by status');
 * results.forEach(r => console.log(r.score, r.chunk_id, r.chunk));
 */
export async function query(indexDir, queryText, options = {}) {
	const {
		topK = 5,
		model = "nomic-embed-text",
		baseUrl = "http://localhost:11434",
	} = options;

	const indexPath = path.join(indexDir, "index.bin");
	const metadataPath = path.join(indexDir, "index_metadata.json");

	if (!fs.existsSync(indexPath)) {
		throw new Error(`Index not found: ${indexPath}`);
	}

	if (!fs.existsSync(metadataPath)) {
		throw new Error(`Metadata not found: ${metadataPath}`);
	}

	const indexer = new FaissIndexer({
		indexType: "IP",
		model,
		baseUrl,
	});

	await indexer.load(indexPath, metadataPath);
	return indexer.search(queryText, topK);
}

/**
 * Build a FAISS index from documents
 *
 * @param {string} inputDir - Directory containing documents
 * @param {string} outputDir - Output directory for index files
 * @param {Object} [options]
 * @param {number} [options.chunkSize=1500] - Chunk size
 * @param {number} [options.chunkOverlap=200] - Overlap between chunks
 * @param {string[]} [options.extensions=['.txt','.md']] - File extensions
 * @param {boolean} [options.recursive=false] - Scan recursively
 * @param {string} [options.model='nomic-embed-text'] - Embedding model
 * @param {string} [options.splitter='recursive'] - Splitter type (recursive, character, markdown, code)
 * @param {Function} [options.onProgress] - Progress callback(current, total)
 * @returns {Promise<{vectors, time}>}
 *
 * @example
 * const result = await build('./documents', './output', { chunkSize: 1000 });
 * console.log(`Built ${result.vectors} vectors in ${result.time}ms`);
 */
export async function build(inputDir, outputDir, options = {}) {
	const {
		chunkSize = 1500,
		chunkOverlap = 200,
		extensions = [".txt", ".md"],
		recursive = false,
		model = "nomic-embed-text",
		baseUrl = "http://localhost:11434",
		splitter = "recursive",
		onProgress,
	} = options;

	// Create output directory
	if (!fs.existsSync(outputDir)) {
		fs.mkdirSync(outputDir, { recursive: true });
	}

	// Generate metadata
	const metadata = [];
	const files = getFiles(inputDir, extensions, recursive);

	for (const filePath of files) {
		const relativePath = path.relative(inputDir, filePath);
		const content = fs.readFileSync(filePath, "utf-8");

		// Use LangChain text splitter
		const chunks = await splitTextByFileType(content, filePath, {
			chunkSize,
			chunkOverlap,
			type: splitter,
		});

		for (let i = 0; i < chunks.length; i++) {
			const baseId = relativePath
				.replace(/[\/\\]/g, "_")
				.replace(/\.[^/.]+$/, "");
			metadata.push({
				doc: relativePath,
				chunk: chunks[i],
				chunk_id: `${baseId}_${i}`,
			});
		}
	}

	// Save metadata
	const metadataPath = path.join(outputDir, "metadata.json");
	fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));

	// Build index
	const indexer = new FaissIndexer({
		indexType: "IP",
		model,
		baseUrl,
	});

	const indexPath = path.join(outputDir, "index.bin");
	return indexer.build(metadataPath, indexPath, onProgress);
}

// Helper: get files from directory
function getFiles(dir, extensions, recursive) {
	const files = [];
	const items = fs.readdirSync(dir, { withFileTypes: true });

	for (const item of items) {
		const fullPath = path.join(dir, item.name);

		if (item.isDirectory() && recursive) {
			files.push(...getFiles(fullPath, extensions, recursive));
		} else if (item.isFile()) {
			const ext = path.extname(item.name).toLowerCase();
			if (extensions.includes(ext)) {
				files.push(fullPath);
			}
		}
	}

	return files;
}
