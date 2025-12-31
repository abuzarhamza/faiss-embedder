/**
 * Text Splitter using LangChain
 *
 * Uses @langchain/textsplitters for intelligent text chunking with support for:
 * - Character-based splitting
 * - Recursive character splitting (respects paragraphs, sentences, words)
 * - Markdown-aware splitting
 * - Code-aware splitting
 */

import {
	RecursiveCharacterTextSplitter,
	CharacterTextSplitter,
	MarkdownTextSplitter,
} from "@langchain/textsplitters";
import path from "path";

/**
 * Available splitter types
 */
export const SPLITTER_TYPES = {
	RECURSIVE: "recursive", // Best for general text, respects structure
	CHARACTER: "character", // Simple character-based split
	MARKDOWN: "markdown", // Markdown-aware splitting
	CODE: "code", // Code-aware splitting with language support
};

/**
 * Language-specific separators for code splitting
 */
const CODE_LANGUAGES = {
	js: "js",
	javascript: "js",
	ts: "ts",
	typescript: "ts",
	py: "python",
	python: "python",
	java: "java",
	go: "go",
	rust: "rust",
	cpp: "cpp",
	c: "cpp",
};

/**
 * Create a text splitter based on options
 *
 * @param {Object} options
 * @param {string} [options.type='recursive'] - Splitter type
 * @param {number} [options.chunkSize=1500] - Target chunk size
 * @param {number} [options.chunkOverlap=200] - Overlap between chunks
 * @param {string} [options.language] - Programming language for code splitting
 * @returns {TextSplitter}
 */
export function createSplitter(options = {}) {
	const {
		type = SPLITTER_TYPES.RECURSIVE,
		chunkSize = 1500,
		chunkOverlap = 200,
		language,
	} = options;

	const baseConfig = {
		chunkSize,
		chunkOverlap,
	};

	switch (type) {
		case SPLITTER_TYPES.CHARACTER:
			return new CharacterTextSplitter({
				...baseConfig,
				separator: "\n\n",
			});

		case SPLITTER_TYPES.MARKDOWN:
			return new MarkdownTextSplitter(baseConfig);

		case SPLITTER_TYPES.CODE:
			if (language && CODE_LANGUAGES[language]) {
				return RecursiveCharacterTextSplitter.fromLanguage(
					CODE_LANGUAGES[language],
					baseConfig
				);
			}
			// Fallback to recursive for unknown languages
			return new RecursiveCharacterTextSplitter(baseConfig);

		case SPLITTER_TYPES.RECURSIVE:
		default:
			return new RecursiveCharacterTextSplitter({
				...baseConfig,
				separators: ["\n\n", "\n", ". ", " ", ""],
			});
	}
}

/**
 * Split text into chunks
 *
 * @param {string} text - Text to split
 * @param {Object} [options]
 * @param {number} [options.chunkSize=1500] - Target chunk size
 * @param {number} [options.chunkOverlap=200] - Overlap between chunks
 * @param {string} [options.type='recursive'] - Splitter type
 * @param {string} [options.language] - Programming language
 * @returns {Promise<string[]>} - Array of text chunks
 *
 * @example
 * const chunks = await splitText(content, { chunkSize: 1000 });
 */
export async function splitText(text, options = {}) {
	const splitter = createSplitter(options);
	return splitter.splitText(text);
}

/**
 * Split text based on file extension
 * Automatically selects the best splitter for the file type
 *
 * @param {string} text - Text to split
 * @param {string} filePath - File path (used to detect type)
 * @param {Object} [options]
 * @param {number} [options.chunkSize=1500] - Target chunk size
 * @param {number} [options.chunkOverlap=200] - Overlap between chunks
 * @returns {Promise<string[]>} - Array of text chunks
 *
 * @example
 * const chunks = await splitTextByFileType(content, 'src/app.js');
 */
export async function splitTextByFileType(text, filePath, options = {}) {
	const ext = path.extname(filePath).toLowerCase().slice(1);

	// Determine splitter type based on extension
	let type = SPLITTER_TYPES.RECURSIVE;
	let language;

	if (ext === "md" || ext === "markdown") {
		type = SPLITTER_TYPES.MARKDOWN;
	} else if (CODE_LANGUAGES[ext]) {
		type = SPLITTER_TYPES.CODE;
		language = ext;
	}

	return splitText(text, { ...options, type, language });
}

/**
 * Legacy compatible chunk function
 * Drop-in replacement for the old chunkText function
 *
 * @param {string} text - Text to split
 * @param {number} [chunkSize=1500] - Target chunk size
 * @param {number} [overlap=200] - Overlap between chunks
 * @returns {Promise<string[]>} - Array of text chunks
 */
export async function chunkText(text, chunkSize = 1500, overlap = 200) {
	return splitText(text, {
		chunkSize,
		chunkOverlap: overlap,
		type: SPLITTER_TYPES.RECURSIVE,
	});
}

/**
 * Synchronous chunk function (for backward compatibility)
 * Uses simple character-based splitting
 *
 * @param {string} text - Text to split
 * @param {number} [chunkSize=1500] - Target chunk size
 * @param {number} [overlap=200] - Overlap between chunks
 * @returns {string[]} - Array of text chunks
 */
export function chunkTextSync(text, chunkSize = 1500, overlap = 200) {
	const chunks = [];
	let start = 0;

	while (start < text.length) {
		const end = Math.min(start + chunkSize, text.length);
		chunks.push(text.slice(start, end));
		start = end - overlap;
		if (start + overlap >= text.length) break;
	}

	return chunks;
}

export default {
	splitText,
	splitTextByFileType,
	chunkText,
	chunkTextSync,
	createSplitter,
	SPLITTER_TYPES,
};
