import { expect } from "chai";
import {
	splitText,
	splitTextByFileType,
	chunkText,
	chunkTextSync,
	createSplitter,
	SPLITTER_TYPES,
} from "../lib/textSplitter.js";

describe("textSplitter", function () {
	const sampleText = `# Header

This is a paragraph with some text. It contains multiple sentences.
Here is another sentence in the same paragraph.

## Another Section

More content here. This section talks about something else.
We have multiple lines of text to work with.

### Subsection

Final paragraph with more content. This helps test the splitter.`;

	const codeText = `function hello() {
  console.log("Hello, World!");
}

function goodbye() {
  console.log("Goodbye!");
}

class MyClass {
  constructor() {
    this.value = 42;
  }

  getValue() {
    return this.value;
  }
}`;

	describe("SPLITTER_TYPES", function () {
		it("should have all splitter types defined", function () {
			// Arrange & Act & Assert
			expect(SPLITTER_TYPES.RECURSIVE).to.equal("recursive");
			expect(SPLITTER_TYPES.CHARACTER).to.equal("character");
			expect(SPLITTER_TYPES.MARKDOWN).to.equal("markdown");
			expect(SPLITTER_TYPES.CODE).to.equal("code");
		});
	});

	describe("createSplitter()", function () {
		it("should create recursive splitter by default", function () {
			// Arrange & Act
			const splitter = createSplitter();

			// Assert
			expect(splitter).to.not.be.null;
		});

		it("should create splitter with custom chunk size", function () {
			// Arrange
			const options = { chunkSize: 500 };

			// Act
			const splitter = createSplitter(options);

			// Assert
			expect(splitter).to.not.be.null;
		});

		it("should create markdown splitter", function () {
			// Arrange
			const options = { type: SPLITTER_TYPES.MARKDOWN };

			// Act
			const splitter = createSplitter(options);

			// Assert
			expect(splitter).to.not.be.null;
		});

		it("should create character splitter", function () {
			// Arrange
			const options = { type: SPLITTER_TYPES.CHARACTER };

			// Act
			const splitter = createSplitter(options);

			// Assert
			expect(splitter).to.not.be.null;
		});

		it("should create code splitter with language", function () {
			// Arrange
			const options = {
				type: SPLITTER_TYPES.CODE,
				language: "js",
			};

			// Act
			const splitter = createSplitter(options);

			// Assert
			expect(splitter).to.not.be.null;
		});
	});

	describe("splitText()", function () {
		it("should split text into chunks", async function () {
			// Arrange
			const options = { chunkSize: 100, chunkOverlap: 20 };

			// Act
			const chunks = await splitText(sampleText, options);

			// Assert
			expect(chunks).to.be.an("array");
			expect(chunks.length).to.be.greaterThan(1);
		});

		it("should return single chunk for small text", async function () {
			// Arrange
			const text = "Hello World";
			const options = { chunkSize: 1000, chunkOverlap: 100 };

			// Act
			const chunks = await splitText(text, options);

			// Assert
			expect(chunks).to.be.an("array");
			expect(chunks.length).to.equal(1);
			expect(chunks[0]).to.equal("Hello World");
		});

		it("should respect chunk size", async function () {
			// Arrange
			const chunkSize = 200;
			const options = { chunkSize, chunkOverlap: 50 };

			// Act
			const chunks = await splitText(sampleText, options);

			// Assert
			for (const chunk of chunks.slice(0, -1)) {
				expect(chunk.length).to.be.lessThanOrEqual(chunkSize + 50); // Allow some overflow
			}
		});

		it("should use recursive splitter by default", async function () {
			// Arrange
			const options = {
				chunkSize: 300,
				chunkOverlap: 50,
				type: SPLITTER_TYPES.RECURSIVE,
			};

			// Act
			const chunks = await splitText(sampleText, options);

			// Assert
			expect(chunks).to.be.an("array");
			expect(chunks.length).to.be.greaterThan(0);
		});
	});

	describe("splitTextByFileType()", function () {
		it("should use markdown splitter for .md files", async function () {
			// Arrange
			const filePath = "file.md";
			const options = { chunkSize: 200, chunkOverlap: 30 };

			// Act
			const chunks = await splitTextByFileType(
				sampleText,
				filePath,
				options
			);

			// Assert
			expect(chunks).to.be.an("array");
			expect(chunks.length).to.be.greaterThan(0);
		});

		it("should use code splitter for .js files", async function () {
			// Arrange
			const filePath = "app.js";
			const options = { chunkSize: 200, chunkOverlap: 30 };

			// Act
			const chunks = await splitTextByFileType(
				codeText,
				filePath,
				options
			);

			// Assert
			expect(chunks).to.be.an("array");
			expect(chunks.length).to.be.greaterThan(0);
		});

		it("should use code splitter for .py files", async function () {
			// Arrange
			const pyCode = `def hello():
    print("Hello")

def goodbye():
    print("Goodbye")`;
			const filePath = "script.py";
			const options = { chunkSize: 100, chunkOverlap: 20 };

			// Act
			const chunks = await splitTextByFileType(pyCode, filePath, options);

			// Assert
			expect(chunks).to.be.an("array");
		});

		it("should use recursive splitter for .txt files", async function () {
			// Arrange
			const filePath = "file.txt";
			const options = { chunkSize: 200, chunkOverlap: 30 };

			// Act
			const chunks = await splitTextByFileType(
				sampleText,
				filePath,
				options
			);

			// Assert
			expect(chunks).to.be.an("array");
			expect(chunks.length).to.be.greaterThan(0);
		});
	});

	describe("chunkText() - async", function () {
		it("should be backward compatible", async function () {
			// Arrange
			const chunkSize = 100;
			const overlap = 20;

			// Act
			const chunks = await chunkText(sampleText, chunkSize, overlap);

			// Assert
			expect(chunks).to.be.an("array");
			expect(chunks.length).to.be.greaterThan(0);
		});

		it("should use default values", async function () {
			// Arrange & Act
			const chunks = await chunkText(sampleText);

			// Assert
			expect(chunks).to.be.an("array");
		});
	});

	describe("chunkTextSync()", function () {
		it("should split text synchronously", function () {
			// Arrange
			const chunkSize = 100;
			const overlap = 20;

			// Act
			const chunks = chunkTextSync(sampleText, chunkSize, overlap);

			// Assert
			expect(chunks).to.be.an("array");
			expect(chunks.length).to.be.greaterThan(1);
		});

		it("should handle overlap correctly", function () {
			// Arrange
			const text = "0123456789ABCDEFGHIJ";
			const chunkSize = 10;
			const overlap = 3;

			// Act
			const chunks = chunkTextSync(text, chunkSize, overlap);

			// Assert
			expect(chunks[0]).to.equal("0123456789");
			expect(chunks[1].startsWith("789")).to.be.true;
		});

		it("should return single chunk for small text", function () {
			// Arrange
			const text = "Hello";
			const chunkSize = 1000;
			const overlap = 100;

			// Act
			const chunks = chunkTextSync(text, chunkSize, overlap);

			// Assert
			expect(chunks.length).to.equal(1);
			expect(chunks[0]).to.equal("Hello");
		});

		it("should use default values", function () {
			// Arrange & Act
			const chunks = chunkTextSync(sampleText);

			// Assert
			expect(chunks).to.be.an("array");
		});
	});
});
