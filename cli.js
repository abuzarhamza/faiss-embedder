#!/usr/bin/env node

/**
 * FAISS Index Generator CLI
 *
 * Generates FAISS index files from a directory of documents.
 *
 * Usage:
 *   npx faiss-gen <input-dir> [output-dir] [options]
 *
 * Examples:
 *   npx faiss-gen ./documents
 *   npx faiss-gen ./documents ./output
 *   npx faiss-gen ./documents --chunk-size 1000 --extensions .md,.txt
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import { generateDocCache } from "./lib/docCache.js";
import { FaissIndexer } from "./lib/faissIndexer.js";
import { OllamaEmbedder } from "./lib/embedder.js";
import { splitTextByFileType, SPLITTER_TYPES } from "./lib/textSplitter.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ═══════════════════════════════════════════════════════════
// Available Embedding Models
// ═══════════════════════════════════════════════════════════
const EMBEDDING_MODELS = {
	"nomic-embed-text": {
		name: "nomic-embed-text",
		provider: "Ollama",
		dimension: 768,
		context: 8192,
		description: "Fast, general-purpose text embeddings",
		default: true,
	},
	"mxbai-embed-large": {
		name: "mxbai-embed-large",
		provider: "Ollama",
		dimension: 1024,
		context: 512,
		description: "High-quality embeddings, larger dimension",
	},
	"all-minilm": {
		name: "all-minilm",
		provider: "Ollama",
		dimension: 384,
		context: 256,
		description: "Lightweight, fast, smaller dimension",
	},
	"snowflake-arctic-embed": {
		name: "snowflake-arctic-embed",
		provider: "Ollama",
		dimension: 1024,
		context: 512,
		description: "Strong retrieval performance",
	},
	"bge-m3": {
		name: "bge-m3",
		provider: "Ollama",
		dimension: 1024,
		context: 8192,
		description: "Multilingual, long context",
	},
};

const DEFAULT_CONFIG = {
	model: "nomic-embed-text",
	baseUrl: "http://localhost:11434",
	chunkSize: 1500,
	overlap: 200,
	indexType: "IP",
	extensions: [".txt", ".md"],
};

/**
 * Show configuration and available models
 */
async function showConfig(checkOllama = false) {
	console.log("\n" + "═".repeat(70));
	console.log("  FAISS Index Generator - Configuration");
	console.log("═".repeat(70));

	// Default Settings
	console.log("\n📋 DEFAULT SETTINGS:");
	console.log("─".repeat(70));
	console.log(`  Embedding Model:   ${DEFAULT_CONFIG.model}`);
	console.log(`  Ollama URL:        ${DEFAULT_CONFIG.baseUrl}`);
	console.log(`  Chunk Size:        ${DEFAULT_CONFIG.chunkSize} characters`);
	console.log(`  Overlap:           ${DEFAULT_CONFIG.overlap} characters`);
	console.log(
		`  Index Type:        ${DEFAULT_CONFIG.indexType} (Inner Product / Cosine)`
	);
	console.log(
		`  Extensions:        ${DEFAULT_CONFIG.extensions.join(
			", "
		)}  (code files supported)`
	);

	// Current Model Details
	const currentModel = EMBEDDING_MODELS[DEFAULT_CONFIG.model];
	console.log("\n🔧 CURRENT EMBEDDING MODEL:");
	console.log("─".repeat(70));
	console.log(`  Name:        ${currentModel.name}`);
	console.log(`  Provider:    ${currentModel.provider}`);
	console.log(`  Dimension:   ${currentModel.dimension}`);
	console.log(`  Context:     ${currentModel.context} tokens`);
	console.log(`  Description: ${currentModel.description}`);

	// Available Models
	console.log("\n📦 AVAILABLE EMBEDDING MODELS (Ollama):");
	console.log("─".repeat(70));
	console.log(
		"  " +
			"Model".padEnd(25) +
			"Dim".padEnd(8) +
			"Context".padEnd(10) +
			"Description"
	);
	console.log("  " + "─".repeat(66));

	for (const [key, model] of Object.entries(EMBEDDING_MODELS)) {
		const isDefault = model.default ? " ⭐" : "";
		console.log(
			"  " +
				(model.name + isDefault).padEnd(25) +
				String(model.dimension).padEnd(8) +
				String(model.context).padEnd(10) +
				model.description
		);
	}

	// How to use different model
	console.log("\n💡 TO USE A DIFFERENT MODEL:");
	console.log("─".repeat(70));
	console.log("  1. Pull the model:     ollama pull mxbai-embed-large");
	console.log(
		"  2. Run with --model:   faiss-gen ./docs --model mxbai-embed-large"
	);

	// Check Ollama status
	if (checkOllama) {
		console.log("\n🔍 OLLAMA STATUS:");
		console.log("─".repeat(70));
		const embedder = new OllamaEmbedder();
		const health = await embedder.healthCheck();
		if (health.ok) {
			console.log(`  ✅ ${health.message}`);
		} else {
			console.log(`  ❌ ${health.message}`);
			console.log("\n  To fix:");
			console.log("    1. Start Ollama:  ollama serve");
			console.log("    2. Pull model:    ollama pull nomic-embed-text");
		}
	}

	console.log("\n" + "═".repeat(70) + "\n");
}

/**
 * Run query against FAISS index
 */
async function runQuery(argv) {
	const indexDir = argv["index-dir"];
	const query = argv.query;
	const topK = argv["top-k"];
	const showChunk = argv["show-chunk"];
	const maxLength = argv["max-length"];
	const model = argv.model || "nomic-embed-text";
	const ollamaUrl = argv["ollama-url"] || "http://localhost:11434";

	// Validate index directory
	const indexPath = path.join(indexDir, "index.bin");
	const metadataPath = path.join(indexDir, "index_metadata.json");

	if (!fs.existsSync(indexPath)) {
		console.error(`❌ Index not found: ${indexPath}`);
		console.error(`   Run: node cli.js build <input-dir> ${indexDir}`);
		process.exit(1);
	}

	if (!fs.existsSync(metadataPath)) {
		console.error(`❌ Metadata not found: ${metadataPath}`);
		process.exit(1);
	}

	console.log("\n" + "═".repeat(70));
	console.log("  FAISS Query Search");
	console.log("═".repeat(70));
	console.log(`  Index:     ${indexDir}`);
	console.log(`  Query:     "${query}"`);
	console.log(`  Top K:     ${topK}`);
	console.log(`  Model:     ${model}`);
	console.log("═".repeat(70) + "\n");

	try {
		// Load and search index
		const indexer = new FaissIndexer({
			indexType: "IP",
			model: model,
			baseUrl: ollamaUrl,
		});

		await indexer.load(indexPath, metadataPath);

		const stats = indexer.getStats();
		console.log(
			`📊 Index loaded: ${stats.vectors} vectors, ${stats.dimension} dimensions\n`
		);

		console.log("🔍 Searching...\n");
		const startTime = Date.now();
		const results = await indexer.search(query, topK);
		const searchTime = Date.now() - startTime;

		console.log(`⏱️  Search completed in ${searchTime}ms\n`);
		console.log("─".repeat(70));

		if (results.length === 0) {
			console.log("  No results found.");
		} else {
			results.forEach((result, i) => {
				const scorePercent = (result.score * 100).toFixed(1);

				console.log(`\n📄 Result ${i + 1}/${results.length}`);
				console.log("─".repeat(70));
				console.log(
					`  Score:     ${result.score.toFixed(
						4
					)} (${scorePercent}% match)`
				);
				console.log(`  Doc:       ${result.doc}`);
				console.log(`  Chunk ID:  ${result.chunk_id}`);

				if (showChunk && result.chunk) {
					let chunkText = result.chunk;

					// Truncate if needed
					if (maxLength > 0 && chunkText.length > maxLength) {
						chunkText =
							chunkText.substring(0, maxLength) +
							"... [truncated]";
					}

					console.log(`\n  Chunk Content:`);
					console.log("  ┌" + "─".repeat(66) + "┐");

					// Format chunk with line breaks
					const lines = chunkText.split("\n");
					for (const line of lines) {
						// Wrap long lines
						const wrappedLines = wrapText(line, 64);
						for (const wLine of wrappedLines) {
							console.log(`  │ ${wLine.padEnd(64)} │`);
						}
					}

					console.log("  └" + "─".repeat(66) + "┘");
				}
			});
		}

		console.log("\n" + "═".repeat(70) + "\n");
	} catch (error) {
		console.error(`\n❌ Query failed: ${error.message}`);

		if (error.message.includes("Ollama")) {
			console.error("\n💡 Make sure Ollama is running:");
			console.error("   ollama serve");
			console.error(`   ollama pull ${model}`);
		}

		process.exit(1);
	}
}

/**
 * Wrap text to specified width
 */
function wrapText(text, width) {
	if (text.length <= width) return [text];

	const lines = [];
	let remaining = text;

	while (remaining.length > width) {
		let breakPoint = remaining.lastIndexOf(" ", width);
		if (breakPoint === -1 || breakPoint < width / 2) {
			breakPoint = width;
		}
		lines.push(remaining.substring(0, breakPoint));
		remaining = remaining.substring(breakPoint).trimStart();
	}

	if (remaining) lines.push(remaining);
	return lines;
}

// Parse arguments with yargs
const argv = yargs(hideBin(process.argv))
	.scriptName("faiss-gen")
	.usage("Usage: $0 <command> [options]")
	.command(
		"config",
		"Show default settings and available embedding models",
		(yargs) => {
			yargs.option("check", {
				type: "boolean",
				description: "Check Ollama connection status",
				default: false,
			});
		},
		async (argv) => {
			await showConfig(argv.check);
			process.exit(0);
		}
	)
	.command(
		"query <index-dir> <query>",
		"Search the FAISS index with a query",
		(yargs) => {
			yargs
				.positional("index-dir", {
					describe: "Directory containing the FAISS index files",
					type: "string",
					demandOption: true,
				})
				.positional("query", {
					describe: "Search query text",
					type: "string",
					demandOption: true,
				})
				.option("top-k", {
					alias: "k",
					type: "number",
					description: "Number of results to return",
					default: 5,
				})
				.option("show-chunk", {
					type: "boolean",
					description: "Show full chunk content",
					default: true,
				})
				.option("max-length", {
					type: "number",
					description:
						"Max characters to show per chunk (0 = no limit)",
					default: 500,
				});
		},
		async (argv) => {
			await runQuery(argv);
			process.exit(0);
		}
	)
	.command(
		["build <input-dir> [output-dir]", "$0 <input-dir> [output-dir]"],
		"Generate FAISS index from documents",
		(yargs) => {
			yargs
				.positional("input-dir", {
					describe: "Directory containing documents to index",
					type: "string",
				})
				.positional("output-dir", {
					describe: "Output directory for index files",
					type: "string",
					default: "./faiss_output",
				});
		},
		async (argv) => {
			// Only run main() if input-dir is provided
			if (argv["input-dir"]) {
				await runBuild(argv);
			}
		}
	)
	.option("chunk-size", {
		alias: "c",
		type: "number",
		description: "Chunk size in characters",
		default: 1500,
	})
	.option("overlap", {
		alias: "o",
		type: "number",
		description: "Overlap between chunks",
		default: 200,
	})
	.option("extensions", {
		alias: "e",
		type: "string",
		description: "Comma-separated file extensions",
		default: ".txt,.md,.js,.json",
		coerce: (val) => val.split(",").map((e) => e.trim()),
	})
	.option("recursive", {
		alias: "r",
		type: "boolean",
		description: "Scan directories recursively",
		default: false,
	})
	.option("index-type", {
		alias: "t",
		type: "string",
		description: "FAISS index type (IP for cosine, L2 for euclidean)",
		choices: ["IP", "L2"],
		default: "IP",
	})
	.option("model", {
		alias: "m",
		type: "string",
		description: "Ollama embedding model to use",
		default: "nomic-embed-text",
		choices: Object.keys(EMBEDDING_MODELS),
	})
	.option("ollama-url", {
		type: "string",
		description: "Ollama server URL",
		default: "http://localhost:11434",
	})
	.option("splitter", {
		alias: "s",
		type: "string",
		description:
			"Text splitter type (recursive, character, markdown, code)",
		choices: ["recursive", "character", "markdown", "code"],
		default: "recursive",
	})
	.option("verbose", {
		alias: "v",
		type: "boolean",
		description: "Show verbose output",
		default: false,
	})
	.example("$0 config", "Show default settings and available models")
	.example("$0 config --check", "Check Ollama connection status")
	.example("$0 query ./faiss_output 'find orders'", "Search index with query")
	.example(
		"$0 query ./faiss_output 'user login' -k 10",
		"Return top 10 results"
	)
	.example("$0 ./documents", "Index documents in ./documents")
	.example("$0 ./docs ./output -c 1000", "Custom chunk size of 1000")
	.example("$0 ./src -e .js,.ts,.py", "Index source code files")
	.example("$0 ./data -r", "Recursively scan subdirectories")
	.example("$0 ./docs -m mxbai-embed-large", "Use different embedding model")
	.epilogue(
		"Output Files:\n" +
			"  doc_index_cache.json    MD5 hashes for change detection\n" +
			"  index_metadata.json     Chunk metadata (doc, chunk, chunk_id)\n" +
			"  index.bin               FAISS binary index"
	)
	.help()
	.alias("help", "h")
	.version("1.0.0")
	.wrap(80)
	.strict()
	.parse();

// Text chunking now uses @langchain/textsplitters via lib/textSplitter.js

/**
 * Read file content
 */
function readFileContent(filePath) {
	const ext = path.extname(filePath).toLowerCase();
	const supportedExtensions = [
		".txt",
		".md",
		".json",
		".js",
		".ts",
		".py",
		".yaml",
		".yml",
		".jsx",
		".tsx",
		".css",
		".html",
		".xml",
		".sql",
		".sh",
		".go",
		".rs",
		".java",
		".c",
		".cpp",
		".h",
		".hpp",
	];
	if (supportedExtensions.includes(ext)) {
		return fs.readFileSync(filePath, "utf-8");
	}
	return null;
}

/**
 * Get files from directory (optionally recursive)
 */
function getFiles(dir, extensions, recursive = false) {
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

/**
 * Run build command
 */
async function runBuild(argv) {
	const options = {
		inputDir: argv["input-dir"],
		outputDir: argv["output-dir"],
		chunkSize: argv["chunk-size"],
		overlap: argv.overlap,
		extensions: argv.extensions,
		recursive: argv.recursive,
		indexType: argv["index-type"],
		model: argv.model,
		ollamaUrl: argv["ollama-url"],
		splitter: argv.splitter,
		verbose: argv.verbose,
	};

	// Validate input directory
	if (!fs.existsSync(options.inputDir)) {
		console.error(`❌ Error: Directory not found: ${options.inputDir}`);
		process.exit(1);
	}

	// Paths
	const docCachePath = path.join(options.outputDir, "doc_index_cache.json");
	const metadataPath = path.join(options.outputDir, "metadata.json");
	const indexPath = path.join(options.outputDir, "index.bin");
	const indexMetadataPath = path.join(
		options.outputDir,
		"index_metadata.json"
	);

	const modelInfo = EMBEDDING_MODELS[options.model] || { dimension: 768 };

	console.log("\n" + "═".repeat(60));
	console.log("  FAISS Index Generator CLI");
	console.log("═".repeat(60));
	console.log(`  Input:       ${options.inputDir}`);
	console.log(`  Output:      ${options.outputDir}`);
	console.log(`  Chunk Size:  ${options.chunkSize}`);
	console.log(`  Overlap:     ${options.overlap}`);
	console.log(`  Splitter:    ${options.splitter} (LangChain)`);
	console.log(`  Extensions:  ${options.extensions.join(", ")}`);
	console.log(`  Recursive:   ${options.recursive}`);
	console.log(`  Index Type:  ${options.indexType}`);
	console.log("─".repeat(60));
	console.log(`  Model:       ${options.model}`);
	console.log(`  Dimension:   ${modelInfo.dimension}`);
	console.log(`  Ollama URL:  ${options.ollamaUrl}`);
	console.log("═".repeat(60) + "\n");

	// Create output directory
	if (!fs.existsSync(options.outputDir)) {
		fs.mkdirSync(options.outputDir, { recursive: true });
	}

	// ═══════════════════════════════════════════════════════════
	// STEP 1: Generate Document Cache
	// ═══════════════════════════════════════════════════════════
	console.log("📁 [1/3] Generating doc_index_cache.json...\n");

	const cache = await generateDocCache(options.inputDir, docCachePath, {
		extensions: options.extensions,
		recursive: options.recursive,
	});

	const fileCount = Object.keys(cache).length;
	console.log(`   ✅ Found ${fileCount} files`);
	console.log(`   📄 Saved: ${docCachePath}\n`);

	// ═══════════════════════════════════════════════════════════
	// STEP 2: Generate Metadata
	// ═══════════════════════════════════════════════════════════
	console.log("─".repeat(60));
	console.log("📝 [2/3] Generating metadata.json...\n");

	const metadata = [];
	const chunkIds = new Set();

	// Get files (with optional recursion)
	const files = getFiles(
		options.inputDir,
		options.extensions,
		options.recursive
	);

	for (const filePath of files) {
		const fileName = path.basename(filePath);
		const relativePath = path.relative(options.inputDir, filePath);
		const content = readFileContent(filePath);

		if (!content) {
			if (options.verbose) {
				console.log(`   ⚠️  Skipping: ${relativePath}`);
			}
			continue;
		}

		// Use LangChain text splitter
		const chunks = await splitTextByFileType(content, filePath, {
			chunkSize: options.chunkSize,
			chunkOverlap: options.overlap,
			type: options.splitter,
		});
		console.log(`   📄 ${relativePath}: ${chunks.length} chunk(s)`);

		for (let i = 0; i < chunks.length; i++) {
			// Use relative path for chunk_id to avoid conflicts
			const baseId = relativePath
				.replace(/[\/\\]/g, "_")
				.replace(/\.[^/.]+$/, "");
			const chunkId = `${baseId}_${i}`;

			if (chunkIds.has(chunkId)) {
				console.error(`   ❌ Duplicate chunk_id: ${chunkId}`);
				process.exit(1);
			}

			metadata.push({
				doc: relativePath,
				chunk: chunks[i],
				chunk_id: chunkId,
			});

			chunkIds.add(chunkId);
		}
	}

	fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
	console.log(`\n   ✅ Total chunks: ${metadata.length}`);
	console.log(`   📄 Saved: ${metadataPath}\n`);

	if (metadata.length === 0) {
		console.error("❌ No documents found to index");
		process.exit(1);
	}

	// ═══════════════════════════════════════════════════════════
	// STEP 3: Build FAISS Index
	// ═══════════════════════════════════════════════════════════
	console.log("─".repeat(60));
	console.log("🔍 [3/3] Building FAISS index...\n");

	try {
		const indexer = new FaissIndexer({
			indexType: options.indexType,
			model: options.model,
			baseUrl: options.ollamaUrl,
		});

		const result = await indexer.build(
			metadataPath,
			indexPath,
			(current, total) => {
				const pct = Math.floor((current / total) * 100);
				process.stdout.write(
					`\r   [${pct
						.toString()
						.padStart(3)}%] ${current}/${total} vectors`
				);
			}
		);

		console.log("\n");
		console.log(`   ✅ Index built successfully!`);
		console.log(`   📊 Vectors: ${result.vectors}`);
		console.log(`   ⏱️  Time: ${(result.time / 1000).toFixed(2)}s`);
		console.log(`   📄 Saved: ${indexPath}`);
		console.log(`   📄 Saved: ${indexMetadataPath}\n`);
	} catch (error) {
		console.error(`\n   ❌ Failed: ${error.message}`);

		if (error.message.includes("Ollama")) {
			console.error("\n   💡 Make sure Ollama is running:");
			console.error("      ollama serve");
			console.error("      ollama pull nomic-embed-text");
		}

		process.exit(1);
	}

	// Summary
	console.log("═".repeat(60));
	console.log("  ✅ Complete! Generated files:");
	console.log("═".repeat(60));
	console.log(`  1. ${docCachePath}`);
	console.log(`  2. ${indexMetadataPath}`);
	console.log(`  3. ${indexPath}`);
	console.log("═".repeat(60) + "\n");
}
