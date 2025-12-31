import { expect } from "chai";
import sinon from "sinon";
import axios from "axios";
import { OllamaEmbedder } from "../lib/embedder.js";

describe("OllamaEmbedder", function () {
	let embedder;
	let axiosGetStub;
	let axiosPostStub;

	// Mock embedding vector (768 dimensions)
	const mockEmbedding = new Array(768)
		.fill(0)
		.map((_, i) => Math.sin(i) * 0.1);

	beforeEach(function () {
		embedder = new OllamaEmbedder();

		// Stub axios methods
		axiosGetStub = sinon.stub(axios, "get");
		axiosPostStub = sinon.stub(axios, "post");
	});

	afterEach(function () {
		sinon.restore();
	});

	describe("constructor", function () {
		it("should create instance with default options", function () {
			// Arrange & Act
			const emb = new OllamaEmbedder();

			// Assert
			expect(emb.baseUrl).to.equal("http://localhost:11434");
			expect(emb.model).to.equal("nomic-embed-text");
			expect(emb.dimension).to.equal(768);
		});

		it("should accept custom options", function () {
			// Arrange
			const options = {
				baseUrl: "http://custom:8080",
				model: "custom-model",
			};

			// Act
			const emb = new OllamaEmbedder(options);

			// Assert
			expect(emb.baseUrl).to.equal("http://custom:8080");
			expect(emb.model).to.equal("custom-model");
		});

		it("should accept only baseUrl option", function () {
			// Arrange
			const options = { baseUrl: "http://localhost:9999" };

			// Act
			const emb = new OllamaEmbedder(options);

			// Assert
			expect(emb.baseUrl).to.equal("http://localhost:9999");
			expect(emb.model).to.equal("nomic-embed-text"); // Default
		});

		it("should accept only model option", function () {
			// Arrange
			const options = { model: "mxbai-embed-large" };

			// Act
			const emb = new OllamaEmbedder(options);

			// Assert
			expect(emb.model).to.equal("mxbai-embed-large");
			expect(emb.baseUrl).to.equal("http://localhost:11434"); // Default
		});
	});

	describe("healthCheck()", function () {
		it("should return ok:true when Ollama is running with model", async function () {
			// Arrange
			axiosGetStub.resolves({
				data: {
					models: [{ name: "nomic-embed-text:latest" }],
				},
			});

			// Act
			const health = await embedder.healthCheck();

			// Assert
			expect(health.ok).to.be.true;
			expect(health.message).to.include("Ollama ready");
			expect(axiosGetStub.calledOnce).to.be.true;
		});

		it("should return ok:false when model not found", async function () {
			// Arrange
			axiosGetStub.resolves({
				data: {
					models: [{ name: "other-model" }],
				},
			});

			// Act
			const health = await embedder.healthCheck();

			// Assert
			expect(health.ok).to.be.false;
			expect(health.message).to.include("not found");
		});

		it("should return ok:false when Ollama is not running", async function () {
			// Arrange
			axiosGetStub.rejects(new Error("ECONNREFUSED"));

			// Act
			const health = await embedder.healthCheck();

			// Assert
			expect(health.ok).to.be.false;
			expect(health.message).to.include("not running");
		});
	});

	describe("embed()", function () {
		it("should generate embedding for text", async function () {
			// Arrange
			const text = "Hello world";
			axiosPostStub.resolves({
				data: { embedding: mockEmbedding },
			});

			// Act
			const embedding = await embedder.embed(text);

			// Assert
			expect(embedding).to.be.instanceOf(Float32Array);
			expect(embedding.length).to.equal(768);
			expect(axiosPostStub.calledOnce).to.be.true;
		});

		it("should call correct API endpoint", async function () {
			// Arrange
			const text = "Test text";
			axiosPostStub.resolves({
				data: { embedding: mockEmbedding },
			});

			// Act
			await embedder.embed(text);

			// Assert
			const [url, body] = axiosPostStub.firstCall.args;
			expect(url).to.equal("http://localhost:11434/api/embeddings");
			expect(body.model).to.equal("nomic-embed-text");
			expect(body.prompt).to.equal("Test text");
		});

		it("should throw error when Ollama not running", async function () {
			// Arrange
			const error = new Error("Connection refused");
			error.code = "ECONNREFUSED";
			axiosPostStub.rejects(error);

			// Act & Assert
			try {
				await embedder.embed("Test");
				expect.fail("Should have thrown error");
			} catch (err) {
				expect(err.message).to.include("Ollama not running");
			}
		});

		it("should throw error when model not found", async function () {
			// Arrange
			const error = new Error("Not found");
			error.response = { status: 404 };
			axiosPostStub.rejects(error);

			// Act & Assert
			try {
				await embedder.embed("Test");
				expect.fail("Should have thrown error");
			} catch (err) {
				expect(err.message).to.include("not found");
			}
		});

		it("should throw error when response missing embedding", async function () {
			// Arrange
			axiosPostStub.resolves({ data: {} });

			// Act & Assert
			try {
				await embedder.embed("Test");
				expect.fail("Should have thrown error");
			} catch (err) {
				expect(err.message).to.include("missing embedding");
			}
		});
	});

	describe("embedBatch()", function () {
		it("should generate embeddings for multiple texts", async function () {
			// Arrange
			const texts = ["First", "Second", "Third"];
			axiosPostStub.resolves({
				data: { embedding: mockEmbedding },
			});

			// Act
			const embeddings = await embedder.embedBatch(texts);

			// Assert
			expect(embeddings).to.be.an("array");
			expect(embeddings.length).to.equal(3);
			expect(axiosPostStub.callCount).to.equal(3);
		});

		it("should call progress callback", async function () {
			// Arrange
			const texts = ["A", "B"];
			const progressSpy = sinon.spy();
			axiosPostStub.resolves({
				data: { embedding: mockEmbedding },
			});

			// Act
			await embedder.embedBatch(texts, progressSpy);

			// Assert
			expect(progressSpy.callCount).to.equal(2);
			expect(progressSpy.firstCall.args).to.deep.equal([1, 2]);
			expect(progressSpy.secondCall.args).to.deep.equal([2, 2]);
		});
	});

	describe("normalize()", function () {
		it("should normalize vector to unit length", function () {
			// Arrange
			const vec = new Float32Array([3, 4, 0]);

			// Act
			const normalized = embedder.normalize(vec);

			// Assert
			let magnitude = 0;
			for (let i = 0; i < normalized.length; i++) {
				magnitude += normalized[i] * normalized[i];
			}
			magnitude = Math.sqrt(magnitude);
			expect(Math.abs(magnitude - 1.0)).to.be.lessThan(0.0001);
		});

		it("should handle zero vector", function () {
			// Arrange
			const vec = new Float32Array([0, 0, 0]);

			// Act
			const normalized = embedder.normalize(vec);

			// Assert
			expect(normalized[0]).to.equal(0);
			expect(normalized[1]).to.equal(0);
			expect(normalized[2]).to.equal(0);
		});

		it("should return Float32Array", function () {
			// Arrange
			const vec = new Float32Array([1, 2, 3]);

			// Act
			const normalized = embedder.normalize(vec);

			// Assert
			expect(normalized).to.be.instanceOf(Float32Array);
		});

		it("should preserve direction", function () {
			// Arrange
			const vec = new Float32Array([2, 4, 6]);

			// Act
			const normalized = embedder.normalize(vec);

			// Assert
			const ratio1 = normalized[1] / normalized[0];
			const ratio2 = normalized[2] / normalized[0];
			expect(Math.abs(ratio1 - 2)).to.be.lessThan(0.0001);
			expect(Math.abs(ratio2 - 3)).to.be.lessThan(0.0001);
		});
	});
});
