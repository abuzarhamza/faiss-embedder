/**
 * Document Cache Generator
 * Generates MD5 hashes for files to detect changes.
 */

import fs from "fs";
import path from "path";
import crypto from "crypto";

/**
 * Calculate MD5 hash of a file
 * @param {string} filePath - Path to the file
 * @returns {Promise<string>} - MD5 hash
 */
export async function getFileHash(filePath) {
	return new Promise((resolve, reject) => {
		const hash = crypto.createHash("md5");
		const stream = fs.createReadStream(filePath);
		stream.on("data", (data) => hash.update(data));
		stream.on("end", () => resolve(hash.digest("hex")));
		stream.on("error", reject);
	});
}

/**
 * Generate doc_index_cache.json from a directory
 *
 * @param {string} dirPath - Directory to scan
 * @param {string} outputPath - Path to save cache file
 * @param {Object} [options] - Options
 * @param {string[]} [options.extensions] - File extensions to include
 * @param {boolean} [options.recursive=false] - Scan subdirectories
 * @returns {Promise<Object>} - Cache object { filename: hash }
 *
 * @example
 * const cache = await generateDocCache('./documents', './cache.json', {
 *   extensions: ['.pdf', '.txt', '.md']
 * });
 */
export async function generateDocCache(dirPath, outputPath, options = {}) {
	const { extensions, recursive = false } = options;

	if (!fs.existsSync(dirPath)) {
		throw new Error(`Directory not found: ${dirPath}`);
	}

	const files = getFiles(dirPath, { extensions, recursive });
	const cache = {};

	for (const filePath of files) {
		const hash = await getFileHash(filePath);
		const key = path.basename(filePath);
		cache[key] = hash;
	}

	// Save cache
	const outputDir = path.dirname(outputPath);
	if (!fs.existsSync(outputDir)) {
		fs.mkdirSync(outputDir, { recursive: true });
	}
	fs.writeFileSync(outputPath, JSON.stringify(cache, null, 2));

	return cache;
}

/**
 * Load existing cache file
 * @param {string} cachePath - Path to cache file
 * @returns {Object|null} - Cache object or null
 */
export function loadDocCache(cachePath) {
	if (fs.existsSync(cachePath)) {
		return JSON.parse(fs.readFileSync(cachePath, "utf-8"));
	}
	return null;
}

/**
 * Detect changes between files and cached hashes
 * @param {string} dirPath - Directory to scan
 * @param {Object} existingCache - Existing cache object
 * @param {Object} [options] - Options
 * @returns {Promise<{added: string[], modified: string[], removed: string[], unchanged: string[]}>}
 */
export async function detectChanges(dirPath, existingCache, options = {}) {
	const { extensions, recursive = false } = options;

	const files = getFiles(dirPath, { extensions, recursive });
	const result = { added: [], modified: [], removed: [], unchanged: [] };
	const currentFiles = new Set();

	for (const filePath of files) {
		const key = path.basename(filePath);
		currentFiles.add(key);

		const hash = await getFileHash(filePath);

		if (!existingCache[key]) {
			result.added.push(key);
		} else if (existingCache[key] !== hash) {
			result.modified.push(key);
		} else {
			result.unchanged.push(key);
		}
	}

	for (const key of Object.keys(existingCache)) {
		if (!currentFiles.has(key)) {
			result.removed.push(key);
		}
	}

	return result;
}

/**
 * Get files from directory
 */
function getFiles(dirPath, options = {}) {
	const { extensions, recursive = false } = options;
	const files = [];

	function scan(currentPath) {
		const entries = fs.readdirSync(currentPath, { withFileTypes: true });

		for (const entry of entries) {
			if (entry.name.startsWith(".")) continue;

			const fullPath = path.join(currentPath, entry.name);

			if (entry.isDirectory() && recursive) {
				scan(fullPath);
			} else if (entry.isFile()) {
				if (extensions && extensions.length > 0) {
					const ext = path.extname(entry.name).toLowerCase();
					if (!extensions.includes(ext)) continue;
				}
				files.push(fullPath);
			}
		}
	}

	scan(dirPath);
	return files;
}

export default { generateDocCache, loadDocCache, detectChanges, getFileHash };
