import { expect } from "chai";
import sinon from "sinon";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { FaissIndexer, buildIndex } from "../lib/faissIndexer.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEST_OUTPUT = path.join(__dirname, "output");

describe("FaissIndexer", function () {
	let sandbox;

	// Mock embedding vector
	const mockEmbedding = new Float32Array(768).fill(0.1);

	beforeEach(function () {
		sandbox = sinon.createSandbox();

		// Create output directory
		if (!fs.existsSync(TEST_OUTPUT)) {
			fs.mkdirSync(TEST_OUTPUT, { recursive: true });
		}
	});

	afterEach(function () {
		sandbox.restore();

		// Cleanup output
		if (fs.existsSync(TEST_OUTPUT)) {
			fs.rmSync(TEST_OUTPUT, { recursive: true });
		}
	});

	describe("constructor", function () {
		it("should create instance with default options", function () {
			// Arrange & Act
			const indexer = new FaissIndexer();

			// Assert
			expect(indexer.indexType).to.equal("IP");
			expect(indexer.model).to.equal("nomic-embed-text");
			expect(indexer.baseUrl).to.equal("http://localhost:11434");
			expect(indexer.dimension).to.equal(768);
			expect(indexer.index).to.be.null;
			expect(indexer.metadata).to.be.an("array").that.is.empty;
		});

		it("should accept custom index type", function () {
			// Arrange
			const options = { indexType: "L2" };

			// Act
			const indexer = new FaissIndexer(options);

			// Assert
			expect(indexer.indexType).to.equal("L2");
		});

		it("should accept IP index type", function () {
			// Arrange
			const options = { indexType: "IP" };

			// Act
			const indexer = new FaissIndexer(options);

			// Assert
			expect(indexer.indexType).to.equal("IP");
		});

		it("should accept custom model", function () {
			// Arrange
			const options = { model: "mxbai-embed-large" };

			// Act
			const indexer = new FaissIndexer(options);

			// Assert
			expect(indexer.model).to.equal("mxbai-embed-large");
			expect(indexer.dimension).to.equal(1024); // mxbai-embed-large dimension
		});

		it("should accept custom baseUrl", function () {
			// Arrange
			const options = { baseUrl: "http://custom:8080" };

			// Act
			const indexer = new FaissIndexer(options);

			// Assert
			expect(indexer.baseUrl).to.equal("http://custom:8080");
		});

		it("should set correct dimension for different models", function () {
			// Arrange
			const models = {
				"nomic-embed-text": 768,
				"mxbai-embed-large": 1024,
				"all-minilm": 384,
				"snowflake-arctic-embed": 1024,
				"bge-m3": 1024,
			};

			// Act & Assert
			for (const [model, expectedDim] of Object.entries(models)) {
				const indexer = new FaissIndexer({ model });
				expect(indexer.dimension).to.equal(expectedDim);
			}
		});

		it("should use default dimension for unknown models", function () {
			// Arrange
			const options = { model: "unknown-model" };

			// Act
			const indexer = new FaissIndexer(options);

			// Assert
			expect(indexer.dimension).to.equal(768);
		});
	});

	describe("build()", function () {
		it("should build index from metadata", async function () {
			// Arrange
			const metadataPath = path.join(TEST_OUTPUT, "test_metadata.json");
			const metadata = [
				{ doc: "test.txt", chunk: "Hello world", chunk_id: "test_0" },
				{ doc: "test.txt", chunk: "Goodbye world", chunk_id: "test_1" },
			];
			fs.writeFileSync(metadataPath, JSON.stringify(metadata));
			const indexPath = path.join(TEST_OUTPUT, "test_index.bin");
			const indexer = new FaissIndexer();
			sandbox
				.stub(indexer.embedder, "healthCheck")
				.resolves({ ok: true });
			sandbox.stub(indexer.embedder, "embed").resolves(mockEmbedding);
			sandbox.stub(indexer.embedder, "normalize").returns(mockEmbedding);

			// Act
			const result = await indexer.build(metadataPath, indexPath);

			// Assert
			expect(result).to.have.property("vectors");
			expect(result).to.have.property("time");
			expect(result.vectors).to.equal(2);
			expect(fs.existsSync(indexPath)).to.be.true;
		});

		it("should call progress callback", async function () {
			// Arrange
			const metadataPath = path.join(
				TEST_OUTPUT,
				"progress_metadata.json"
			);
			const metadata = [
				{ doc: "a.txt", chunk: "Text A", chunk_id: "a_0" },
				{ doc: "b.txt", chunk: "Text B", chunk_id: "b_0" },
			];
			fs.writeFileSync(metadataPath, JSON.stringify(metadata));
			const indexPath = path.join(TEST_OUTPUT, "progress_index.bin");
			const indexer = new FaissIndexer();
			sandbox
				.stub(indexer.embedder, "healthCheck")
				.resolves({ ok: true });
			sandbox.stub(indexer.embedder, "embed").resolves(mockEmbedding);
			sandbox.stub(indexer.embedder, "normalize").returns(mockEmbedding);
			const progressSpy = sinon.spy();

			// Act
			await indexer.build(metadataPath, indexPath, progressSpy);

			// Assert
			expect(progressSpy.callCount).to.equal(2);
			expect(progressSpy.firstCall.args[0]).to.equal(1);
			expect(progressSpy.firstCall.args[1]).to.equal(2);
		});

		it("should throw error for missing metadata file", async function () {
			// Arrange
			const indexer = new FaissIndexer();
			sandbox
				.stub(indexer.embedder, "healthCheck")
				.resolves({ ok: true });
			const invalidPath = "/non/existent/metadata.json";

			// Act & Assert
			try {
				await indexer.build(invalidPath, "index.bin");
				expect.fail("Should have thrown error");
			} catch (error) {
				expect(error.message).to.include("not found");
			}
		});

		it("should throw error for empty metadata", async function () {
			// Arrange
			const metadataPath = path.join(TEST_OUTPUT, "empty_metadata.json");
			fs.writeFileSync(metadataPath, "[]");
			const indexer = new FaissIndexer();
			sandbox
				.stub(indexer.embedder, "healthCheck")
				.resolves({ ok: true });

			// Act & Assert
			try {
				await indexer.build(metadataPath, "index.bin");
				expect.fail("Should have thrown error");
			} catch (error) {
				expect(error.message).to.include("non-empty");
			}
		});

		it("should throw error when Ollama health check fails", async function () {
			// Arrange
			const metadataPath = path.join(TEST_OUTPUT, "health_metadata.json");
			fs.writeFileSync(
				metadataPath,
				JSON.stringify([{ doc: "a", chunk: "b", chunk_id: "c" }])
			);
			const indexer = new FaissIndexer();
			sandbox.stub(indexer.embedder, "healthCheck").resolves({
				ok: false,
				message: "Ollama not running",
			});

			// Act & Assert
			try {
				await indexer.build(metadataPath, "index.bin");
				expect.fail("Should have thrown error");
			} catch (error) {
				expect(error.message).to.include("Ollama");
			}
		});

		it("should skip empty chunks", async function () {
			// Arrange
			const metadataPath = path.join(TEST_OUTPUT, "empty_chunk.json");
			const metadata = [
				{ doc: "a.txt", chunk: "Valid text", chunk_id: "a_0" },
				{ doc: "b.txt", chunk: "   ", chunk_id: "b_0" }, // Empty
				{ doc: "c.txt", chunk: "", chunk_id: "c_0" }, // Empty
			];
			fs.writeFileSync(metadataPath, JSON.stringify(metadata));
			const indexPath = path.join(TEST_OUTPUT, "skip_empty.bin");
			const indexer = new FaissIndexer();
			sandbox
				.stub(indexer.embedder, "healthCheck")
				.resolves({ ok: true });
			sandbox.stub(indexer.embedder, "embed").resolves(mockEmbedding);
			sandbox.stub(indexer.embedder, "normalize").returns(mockEmbedding);

			// Act
			const result = await indexer.build(metadataPath, indexPath);

			// Assert
			expect(result.vectors).to.equal(1); // Only 1 valid chunk
		});

		it("should save metadata alongside index", async function () {
			// Arrange
			const metadataPath = path.join(TEST_OUTPUT, "meta_save.json");
			fs.writeFileSync(
				metadataPath,
				JSON.stringify([{ doc: "a", chunk: "text", chunk_id: "a_0" }])
			);
			const indexPath = path.join(TEST_OUTPUT, "meta_index.bin");
			const indexer = new FaissIndexer();
			sandbox
				.stub(indexer.embedder, "healthCheck")
				.resolves({ ok: true });
			sandbox.stub(indexer.embedder, "embed").resolves(mockEmbedding);
			sandbox.stub(indexer.embedder, "normalize").returns(mockEmbedding);

			// Act
			await indexer.build(metadataPath, indexPath);

			// Assert
			const metaPath = indexPath.replace(".bin", "_metadata.json");
			expect(fs.existsSync(metaPath)).to.be.true;
		});
	});

	describe("load()", function () {
		it("should load existing index", async function () {
			// Arrange - First build an index
			const metadataPath = path.join(TEST_OUTPUT, "load_metadata.json");
			fs.writeFileSync(
				metadataPath,
				JSON.stringify([
					{ doc: "test", chunk: "text", chunk_id: "t_0" },
				])
			);
			const indexPath = path.join(TEST_OUTPUT, "load_index.bin");
			const indexer1 = new FaissIndexer();
			sandbox
				.stub(indexer1.embedder, "healthCheck")
				.resolves({ ok: true });
			sandbox.stub(indexer1.embedder, "embed").resolves(mockEmbedding);
			sandbox.stub(indexer1.embedder, "normalize").returns(mockEmbedding);
			await indexer1.build(metadataPath, indexPath);
			sandbox.restore(); // Restore to get fresh stubs
			const metaPath = indexPath.replace(".bin", "_metadata.json");

			// Act
			const indexer2 = new FaissIndexer();
			await indexer2.load(indexPath, metaPath);

			// Assert
			expect(indexer2.index).to.not.be.null;
			expect(indexer2.metadata.length).to.equal(1);
		});

		it("should throw error for non-existent index", async function () {
			// Arrange
			const indexer = new FaissIndexer();
			const invalidPath = "/non/existent/index.bin";

			// Act & Assert
			try {
				await indexer.load(invalidPath);
				expect.fail("Should have thrown error");
			} catch (error) {
				expect(error.message).to.include("not found");
			}
		});
	});

	describe("search()", function () {
		it("should throw error for empty index", async function () {
			// Arrange
			const indexer = new FaissIndexer();

			// Act & Assert
			try {
				await indexer.search("test query");
				expect.fail("Should have thrown error");
			} catch (error) {
				expect(error.message).to.include("empty");
			}
		});

		it("should return results with correct structure", async function () {
			// Arrange
			const metadataPath = path.join(TEST_OUTPUT, "search_meta.json");
			fs.writeFileSync(
				metadataPath,
				JSON.stringify([
					{ doc: "a.txt", chunk: "Hello", chunk_id: "a_0" },
					{ doc: "b.txt", chunk: "World", chunk_id: "b_0" },
				])
			);
			const indexPath = path.join(TEST_OUTPUT, "search_index.bin");
			const indexer = new FaissIndexer();
			sandbox
				.stub(indexer.embedder, "healthCheck")
				.resolves({ ok: true });
			sandbox.stub(indexer.embedder, "embed").resolves(mockEmbedding);
			sandbox.stub(indexer.embedder, "normalize").returns(mockEmbedding);
			await indexer.build(metadataPath, indexPath);

			// Act
			const results = await indexer.search("test", 2);

			// Assert
			expect(results).to.be.an("array");
			expect(results[0]).to.have.property("id");
			expect(results[0]).to.have.property("score");
			expect(results[0]).to.have.property("doc");
			expect(results[0]).to.have.property("chunk_id");
			expect(results[0]).to.have.property("chunk");
		});
	});

	describe("getStats()", function () {
		it("should return stats for empty indexer", function () {
			// Arrange & Act
			const indexer = new FaissIndexer();
			const stats = indexer.getStats();

			// Assert
			expect(stats.vectors).to.equal(0);
			expect(stats.dimension).to.equal(768);
			expect(stats.type).to.equal("IP");
			expect(stats.model).to.equal("nomic-embed-text");
			expect(stats.baseUrl).to.equal("http://localhost:11434");
		});

		it("should return correct stats after build", async function () {
			// Arrange
			const metadataPath = path.join(TEST_OUTPUT, "stats_meta.json");
			fs.writeFileSync(
				metadataPath,
				JSON.stringify([{ doc: "a", chunk: "text", chunk_id: "a_0" }])
			);
			const indexPath = path.join(TEST_OUTPUT, "stats_index.bin");
			const indexer = new FaissIndexer();
			sandbox
				.stub(indexer.embedder, "healthCheck")
				.resolves({ ok: true });
			sandbox.stub(indexer.embedder, "embed").resolves(mockEmbedding);
			sandbox.stub(indexer.embedder, "normalize").returns(mockEmbedding);
			await indexer.build(metadataPath, indexPath);

			// Act
			const stats = indexer.getStats();

			// Assert
			expect(stats.vectors).to.equal(1);
			expect(stats.dimension).to.equal(768);
			expect(stats.type).to.equal("IP");
			expect(stats.model).to.equal("nomic-embed-text");
			expect(stats.baseUrl).to.equal("http://localhost:11434");
		});

		it("should return custom model in stats", function () {
			// Arrange
			const options = {
				model: "mxbai-embed-large",
				baseUrl: "http://custom:8080",
			};

			// Act
			const indexer = new FaissIndexer(options);
			const stats = indexer.getStats();

			// Assert
			expect(stats.model).to.equal("mxbai-embed-large");
			expect(stats.baseUrl).to.equal("http://custom:8080");
			expect(stats.dimension).to.equal(1024);
		});
	});

	describe("buildIndex() function", function () {
		it("should be a convenience wrapper", async function () {
			// Arrange
			const metadataPath = path.join(TEST_OUTPUT, "func_metadata.json");
			fs.writeFileSync(
				metadataPath,
				JSON.stringify([
					{ doc: "test", chunk: "text", chunk_id: "t_0" },
				])
			);

			// Act & Assert
			expect(buildIndex).to.be.a("function");
			expect(buildIndex.length).to.be.at.least(2); // At least 2 params
		});

		it("should accept options including model", function () {
			// Arrange & Act & Assert
			expect(buildIndex).to.be.a("function");
		});
	});
});
