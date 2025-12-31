/**
 * Ollama Embedder
 * Uses nomic-embed-text model for generating embeddings.
 */

import axios from "axios";

export class OllamaEmbedder {
	/**
	 * Create an OllamaEmbedder instance
	 * @param {Object} [options]
	 * @param {string} [options.baseUrl='http://localhost:11434'] - Ollama server URL
	 * @param {string} [options.model='nomic-embed-text'] - Embedding model
	 */
	constructor(options = {}) {
		this.baseUrl = options.baseUrl || "http://localhost:11434";
		this.model = options.model || "nomic-embed-text";
		this.dimension = 768;
	}

	/**
	 * Generate embedding for text
	 * @param {string} text
	 * @returns {Promise<Float32Array>}
	 */
	async embed(text) {
		try {
			const response = await axios.post(
				`${this.baseUrl}/api/embeddings`,
				{ model: this.model, prompt: text },
				{ timeout: 30000 }
			);

			if (!response.data?.embedding) {
				throw new Error("Invalid response: missing embedding");
			}

			return new Float32Array(response.data.embedding);
		} catch (error) {
			if (error.code === "ECONNREFUSED") {
				throw new Error(
					`Ollama not running at ${this.baseUrl}. Start with: ollama serve`
				);
			}
			if (error.response?.status === 404) {
				throw new Error(
					`Model '${this.model}' not found. Run: ollama pull ${this.model}`
				);
			}
			throw new Error(`Embedding failed: ${error.message}`);
		}
	}

	/**
	 * Generate embeddings for multiple texts
	 * @param {string[]} texts
	 * @param {Function} [onProgress] - Callback (current, total)
	 * @returns {Promise<Float32Array[]>}
	 */
	async embedBatch(texts, onProgress) {
		const embeddings = [];

		for (let i = 0; i < texts.length; i++) {
			embeddings.push(await this.embed(texts[i]));
			if (onProgress) onProgress(i + 1, texts.length);
		}

		return embeddings;
	}

	/**
	 * Check if Ollama is running
	 * @returns {Promise<{ok: boolean, message: string}>}
	 */
	async healthCheck() {
		try {
			const response = await axios.get(`${this.baseUrl}/api/tags`, {
				timeout: 5000,
			});

			const models = response.data.models || [];
			const hasModel = models.some((m) => m.name.includes(this.model));

			if (!hasModel) {
				return {
					ok: false,
					message: `Model '${this.model}' not found. Run: ollama pull ${this.model}`,
				};
			}

			return { ok: true, message: `Ollama ready with ${this.model}` };
		} catch (error) {
			return {
				ok: false,
				message: `Ollama not running: ${error.message}`,
			};
		}
	}

	/**
	 * Normalize vector for cosine similarity
	 * @param {Float32Array} vec
	 * @returns {Float32Array}
	 */
	normalize(vec) {
		let norm = 0;
		for (let i = 0; i < vec.length; i++) {
			norm += vec[i] * vec[i];
		}
		norm = Math.sqrt(norm);

		if (norm === 0) return vec;

		const result = new Float32Array(vec.length);
		for (let i = 0; i < vec.length; i++) {
			result[i] = vec[i] / norm;
		}
		return result;
	}
}

export default OllamaEmbedder;
