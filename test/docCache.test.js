import { expect } from "chai";
import sinon from "sinon";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
	generateDocCache,
	loadDocCache,
	detectChanges,
	getFileHash,
} from "../lib/docCache.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEST_DIR = path.join(__dirname, "fixtures");
const TEST_OUTPUT = path.join(__dirname, "output");

describe("docCache", function () {
	let sandbox;

	// Setup: Create test fixtures
	before(function () {
		if (!fs.existsSync(TEST_DIR)) {
			fs.mkdirSync(TEST_DIR, { recursive: true });
		}
		if (!fs.existsSync(TEST_OUTPUT)) {
			fs.mkdirSync(TEST_OUTPUT, { recursive: true });
		}

		// Create test files
		fs.writeFileSync(path.join(TEST_DIR, "file1.txt"), "Hello World");
		fs.writeFileSync(path.join(TEST_DIR, "file2.txt"), "Test Content");
		fs.writeFileSync(path.join(TEST_DIR, "file3.md"), "# Markdown");
	});

	beforeEach(function () {
		sandbox = sinon.createSandbox();
	});

	afterEach(function () {
		sandbox.restore();
	});

	// Cleanup
	after(function () {
		if (fs.existsSync(TEST_DIR)) {
			fs.rmSync(TEST_DIR, { recursive: true });
		}
		if (fs.existsSync(TEST_OUTPUT)) {
			fs.rmSync(TEST_OUTPUT, { recursive: true });
		}
	});

	describe("getFileHash()", function () {
		it("should return MD5 hash of file", async function () {
			// Arrange
			const filePath = path.join(TEST_DIR, "file1.txt");

			// Act
			const hash = await getFileHash(filePath);

			// Assert
			expect(hash).to.be.a("string");
			expect(hash).to.have.length(32); // MD5 is 32 hex chars
		});

		it("should return same hash for same content", async function () {
			// Arrange
			const filePath = path.join(TEST_DIR, "file1.txt");

			// Act
			const hash1 = await getFileHash(filePath);
			const hash2 = await getFileHash(filePath);

			// Assert
			expect(hash1).to.equal(hash2);
		});

		it("should return different hash for different content", async function () {
			// Arrange
			const file1Path = path.join(TEST_DIR, "file1.txt");
			const file2Path = path.join(TEST_DIR, "file2.txt");

			// Act
			const hash1 = await getFileHash(file1Path);
			const hash2 = await getFileHash(file2Path);

			// Assert
			expect(hash1).to.not.equal(hash2);
		});

		it("should return consistent hash for known content", async function () {
			// Arrange
			const filePath = path.join(TEST_DIR, "file1.txt");
			const expectedHash = "b10a8db164e0754105b7a99be72e3fe5"; // MD5 of "Hello World"

			// Act
			const hash = await getFileHash(filePath);

			// Assert
			expect(hash).to.equal(expectedHash);
		});
	});

	describe("generateDocCache()", function () {
		it("should generate cache for directory", async function () {
			// Arrange
			const outputPath = path.join(TEST_OUTPUT, "cache1.json");

			// Act
			const cache = await generateDocCache(TEST_DIR, outputPath);

			// Assert
			expect(cache).to.be.an("object");
			expect(Object.keys(cache).length).to.be.greaterThan(0);
		});

		it("should save cache to file", async function () {
			// Arrange
			const outputPath = path.join(TEST_OUTPUT, "cache2.json");

			// Act
			await generateDocCache(TEST_DIR, outputPath);

			// Assert
			expect(fs.existsSync(outputPath)).to.be.true;
			const content = JSON.parse(fs.readFileSync(outputPath, "utf-8"));
			expect(content).to.be.an("object");
		});

		it("should filter by extensions", async function () {
			// Arrange
			const outputPath = path.join(TEST_OUTPUT, "cache3.json");
			const options = { extensions: [".txt"] };

			// Act
			const cache = await generateDocCache(TEST_DIR, outputPath, options);

			// Assert
			const keys = Object.keys(cache);
			expect(keys.every((k) => k.endsWith(".txt"))).to.be.true;
			expect(keys.some((k) => k.endsWith(".md"))).to.be.false;
		});

		it("should include multiple extensions", async function () {
			// Arrange
			const outputPath = path.join(TEST_OUTPUT, "cache4.json");
			const options = { extensions: [".txt", ".md"] };

			// Act
			const cache = await generateDocCache(TEST_DIR, outputPath, options);

			// Assert
			const keys = Object.keys(cache);
			expect(keys.some((k) => k.endsWith(".txt"))).to.be.true;
			expect(keys.some((k) => k.endsWith(".md"))).to.be.true;
		});

		it("should throw error for non-existent directory", async function () {
			// Arrange
			const invalidPath = "/non/existent/path";
			const outputPath = "output.json";

			// Act & Assert
			try {
				await generateDocCache(invalidPath, outputPath);
				expect.fail("Should have thrown error");
			} catch (error) {
				expect(error.message).to.include("not found");
			}
		});

		it("should create output directory if not exists", async function () {
			// Arrange
			const nestedOutput = path.join(
				TEST_OUTPUT,
				"nested",
				"deep",
				"cache.json"
			);

			// Act
			await generateDocCache(TEST_DIR, nestedOutput);

			// Assert
			expect(fs.existsSync(nestedOutput)).to.be.true;
		});
	});

	describe("loadDocCache()", function () {
		it("should load existing cache file", async function () {
			// Arrange
			const outputPath = path.join(TEST_OUTPUT, "cache_load.json");
			const originalCache = await generateDocCache(TEST_DIR, outputPath);

			// Act
			const loadedCache = loadDocCache(outputPath);

			// Assert
			expect(loadedCache).to.deep.equal(originalCache);
		});

		it("should return null for non-existent file", function () {
			// Arrange
			const nonExistentPath = "/non/existent/cache.json";

			// Act
			const result = loadDocCache(nonExistentPath);

			// Assert
			expect(result).to.be.null;
		});

		it("should parse JSON correctly", function () {
			// Arrange
			const testPath = path.join(TEST_OUTPUT, "manual_cache.json");
			const testData = { "file.txt": "abc123" };
			fs.writeFileSync(testPath, JSON.stringify(testData));

			// Act
			const loaded = loadDocCache(testPath);

			// Assert
			expect(loaded).to.deep.equal(testData);
		});
	});

	describe("detectChanges()", function () {
		it("should detect unchanged files", async function () {
			// Arrange
			const outputPath = path.join(TEST_OUTPUT, "cache_detect.json");
			const cache = await generateDocCache(TEST_DIR, outputPath, {
				extensions: [".txt"],
			});

			// Act
			const changes = await detectChanges(TEST_DIR, cache, {
				extensions: [".txt"],
			});

			// Assert
			expect(changes.unchanged.length).to.be.greaterThan(0);
			expect(changes.added.length).to.equal(0);
			expect(changes.modified.length).to.equal(0);
		});

		it("should detect added files", async function () {
			// Arrange
			const outputPath = path.join(TEST_OUTPUT, "cache_add.json");
			const cache = await generateDocCache(TEST_DIR, outputPath, {
				extensions: [".txt"],
			});
			const newFilePath = path.join(TEST_DIR, "new_file.txt");
			fs.writeFileSync(newFilePath, "New");

			// Act
			const changes = await detectChanges(TEST_DIR, cache, {
				extensions: [".txt"],
			});

			// Assert
			expect(changes.added).to.include("new_file.txt");

			// Cleanup
			fs.unlinkSync(newFilePath);
		});

		it("should detect modified files", async function () {
			// Arrange
			const outputPath = path.join(TEST_OUTPUT, "cache_modify.json");
			const cache = await generateDocCache(TEST_DIR, outputPath, {
				extensions: [".txt"],
			});
			const file1Path = path.join(TEST_DIR, "file1.txt");
			const original = fs.readFileSync(file1Path, "utf-8");
			fs.writeFileSync(file1Path, "Modified Content");

			// Act
			const changes = await detectChanges(TEST_DIR, cache, {
				extensions: [".txt"],
			});

			// Assert
			expect(changes.modified).to.include("file1.txt");

			// Cleanup - Restore original
			fs.writeFileSync(file1Path, original);
		});

		it("should detect removed files", async function () {
			// Arrange
			const tempPath = path.join(TEST_DIR, "temp.txt");
			fs.writeFileSync(tempPath, "Temp");
			const outputPath = path.join(TEST_OUTPUT, "cache_remove.json");
			const cache = await generateDocCache(TEST_DIR, outputPath, {
				extensions: [".txt"],
			});
			fs.unlinkSync(tempPath); // Remove file

			// Act
			const changes = await detectChanges(TEST_DIR, cache, {
				extensions: [".txt"],
			});

			// Assert
			expect(changes.removed).to.include("temp.txt");
		});

		it("should return all change types", async function () {
			// Arrange
			const cache = { "existing.txt": "abc123" };

			// Act
			const changes = await detectChanges(TEST_DIR, cache, {
				extensions: [".txt"],
			});

			// Assert
			expect(changes).to.have.property("added");
			expect(changes).to.have.property("modified");
			expect(changes).to.have.property("removed");
			expect(changes).to.have.property("unchanged");
			expect(changes.added).to.be.an("array");
			expect(changes.modified).to.be.an("array");
			expect(changes.removed).to.be.an("array");
			expect(changes.unchanged).to.be.an("array");
		});
	});
});
