/**
 * FAISS Index Generator
 * Generates FAISS index from metadata.json using Ollama embeddings.
 */

import fs from "fs";
import path from "path";
import faiss from "faiss-node";
import { OllamaEmbedder } from "./embedder.js";

const { IndexFlatL2, IndexFlatIP } = faiss;

export class FaissIndexer {
	/**
	 * Create a FaissIndexer instance
	 * @param {Object} [options]
	 * @param {string} [options.indexType='IP'] - 'IP' (cosine) or 'L2' (euclidean)
	 * @param {string} [options.model='nomic-embed-text'] - Ollama embedding model
	 * @param {string} [options.baseUrl='http://localhost:11434'] - Ollama server URL
	 * @param {Object} [options.embedder] - Custom embedder options (overrides model/baseUrl)
	 */
	constructor(options = {}) {
		this.indexType = options.indexType || "IP";
		this.model = options.model || "nomic-embed-text";
		this.baseUrl = options.baseUrl || "http://localhost:11434";

		// Model dimension mapping
		const modelDimensions = {
			"nomic-embed-text": 768,
			"mxbai-embed-large": 1024,
			"all-minilm": 384,
			"snowflake-arctic-embed": 1024,
			"bge-m3": 1024,
		};
		this.dimension = modelDimensions[this.model] || 768;

		// Create embedder with model settings
		this.embedder = new OllamaEmbedder(
			options.embedder || {
				model: this.model,
				baseUrl: this.baseUrl,
			}
		);
		this.index = null;
		this.metadata = [];
	}

	/**
	 * Build FAISS index from metadata.json
	 *
	 * @param {string} metadataPath - Path to metadata.json
	 * @param {string} outputPath - Path to save index.bin
	 * @param {Function} [onProgress] - Callback (current, total, item)
	 * @returns {Promise<{vectors: number, time: number}>}
	 *
	 * @example
	 * const indexer = new FaissIndexer();
	 * await indexer.build('./metadata.json', './index.bin');
	 */
	async build(metadataPath, outputPath, onProgress) {
		// Health check
		const health = await this.embedder.healthCheck();
		if (!health.ok) throw new Error(health.message);

		// Load metadata
		if (!fs.existsSync(metadataPath)) {
			throw new Error(`Metadata file not found: ${metadataPath}`);
		}

		const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
		if (!Array.isArray(metadata) || metadata.length === 0) {
			throw new Error("Metadata must be a non-empty array");
		}

		// Initialize index
		this.index =
			this.indexType === "L2"
				? new IndexFlatL2(this.dimension)
				: new IndexFlatIP(this.dimension);
		this.metadata = [];

		const startTime = Date.now();

		// Process chunks
		for (let i = 0; i < metadata.length; i++) {
			const item = metadata[i];
			const chunk = item.chunk || "";

			if (!chunk.trim()) continue;

			// Generate and normalize embedding
			let embedding = await this.embedder.embed(chunk);
			embedding = this.embedder.normalize(embedding);

			// Add to index (convert Float32Array to Array for faiss-node)
			this.index.add(Array.from(embedding));

			// Store metadata (full chunk, plus any extra fields from input)
			this.metadata.push({
				id: this.metadata.length,
				doc: item.doc,
				chunk_id: item.chunk_id,
				chunk: chunk,
				...Object.fromEntries(
					Object.entries(item).filter(
						([k]) => !["doc", "chunk_id", "chunk"].includes(k)
					)
				),
			});

			if (onProgress) onProgress(i + 1, metadata.length, item);
		}

		// Save index
		this._save(outputPath);

		return {
			vectors: this.index.ntotal(),
			time: Date.now() - startTime,
		};
	}

	/**
	 * Save index to file
	 * @private
	 */
	_save(indexPath) {
		const dir = path.dirname(indexPath);
		if (!fs.existsSync(dir)) {
			fs.mkdirSync(dir, { recursive: true });
		}

		// Save FAISS index
		this.index.write(indexPath);

		// Save metadata
		const metaPath = indexPath.replace(".bin", "_metadata.json");
		fs.writeFileSync(metaPath, JSON.stringify(this.metadata, null, 2));
	}

	/**
	 * Load existing index
	 * @param {string} indexPath - Path to index.bin
	 * @param {string} [metadataPath] - Path to metadata file
	 */
	async load(indexPath, metadataPath) {
		if (!fs.existsSync(indexPath)) {
			throw new Error(`Index not found: ${indexPath}`);
		}

		this.index =
			this.indexType === "L2"
				? IndexFlatL2.read(indexPath)
				: IndexFlatIP.read(indexPath);

		if (metadataPath && fs.existsSync(metadataPath)) {
			this.metadata = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
		}

		return this;
	}

	/**
	 * Search for similar vectors
	 *
	 * @param {string} query - Query text
	 * @param {number} [k=5] - Number of results
	 * @returns {Promise<Array<{id: number, score: number, doc: string, chunk_id: string, chunk: string}>>}
	 */
	async search(query, k = 5) {
		if (!this.index || this.index.ntotal() === 0) {
			throw new Error("Index is empty or not loaded");
		}

		let queryVec = await this.embedder.embed(query);
		queryVec = this.embedder.normalize(queryVec);

		// Convert Float32Array to Array for faiss-node
		const result = this.index.search(
			Array.from(queryVec),
			Math.min(k, this.index.ntotal())
		);

		const results = [];
		for (let i = 0; i < result.labels.length; i++) {
			const idx = result.labels[i];
			if (idx === -1) continue;

			const meta = this.metadata[idx] || { id: idx };
			results.push({
				id: idx,
				score: result.distances[i],
				doc: meta.doc,
				chunk_id: meta.chunk_id,
				chunk: meta.chunk,
			});
		}

		return results;
	}

	/**
	 * Get index stats
	 */
	getStats() {
		return {
			vectors: this.index ? this.index.ntotal() : 0,
			dimension: this.dimension,
			type: this.indexType,
			model: this.model,
			baseUrl: this.baseUrl,
		};
	}
}

/**
 * Build FAISS index from metadata.json
 *
 * @param {string} metadataPath - Path to metadata.json
 * @param {string} outputPath - Path to save index.bin
 * @param {Object} [options]
 * @param {Function} [options.onProgress] - Progress callback
 * @returns {Promise<{vectors: number, time: number}>}
 */
export async function buildIndex(metadataPath, outputPath, options = {}) {
	const indexer = new FaissIndexer(options);
	return indexer.build(metadataPath, outputPath, options.onProgress);
}

export default FaissIndexer;
