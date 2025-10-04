import pytest
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from ragas.metrics import faithfulness, answer_relevancy
from ragas import SingleTurnSample, EvaluationDataset
from ragas.evaluation import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def mock_rag_output_technical():
    return [
        {
            "question": "What is the difference between CUDA cores and Tensor cores?",
            "answer": "CUDA cores are general-purpose parallel processors, while Tensor cores are specialized hardware units designed specifically for matrix multiplication operations used in deep learning.",
            "contexts": ["CUDA cores handle general parallel computing tasks, whereas Tensor cores are specialized processing units optimized for matrix operations in AI workloads. Tensor cores can perform mixed-precision matrix multiply-and-accumulate calculations significantly faster than CUDA cores for deep learning operations."]
        },
        {
            "question": "How does FAISS improve vector search performance?",
            "answer": "FAISS improves vector search performance through efficient indexing algorithms like IVF, PQ, and HNSW that reduce search complexity from linear to sublinear time.",
            "contexts": ["FAISS (Facebook AI Similarity Search) uses advanced indexing structures such as Inverted File Index (IVF), Product Quantization (PQ), and Hierarchical Navigable Small World (HNSW) graphs to accelerate similarity search in high-dimensional vector spaces, reducing computational complexity significantly."]
        },
        {
            "question": "What is the purpose of the attention mechanism in transformers?",
            "answer": "The attention mechanism allows transformers to weigh the importance of different parts of the input sequence when processing each token, enabling the model to capture long-range dependencies.",
            "contexts": ["The attention mechanism in transformer architectures computes relevance scores between all pairs of tokens in a sequence, allowing the model to focus on relevant context regardless of distance. This is achieved through query, key, and value matrices that determine which parts of the input are most important for each output."]
        },
        {
            "question": "What is RAG in the context of large language models?",
            "answer": "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation, allowing LLMs to access external knowledge bases to provide more accurate and up-to-date responses.",
            "contexts": ["Retrieval-Augmented Generation (RAG) enhances language models by first retrieving relevant documents from a knowledge base using semantic search, then using those documents as context for the LLM to generate informed responses. This approach reduces hallucinations and enables models to incorporate current information beyond their training data."]
        },
        {
            "question": "What is the difference between embedding models and generative models?",
            "answer": "Embedding models convert text into dense vector representations for similarity comparison, while generative models produce new text based on input prompts.",
            "contexts": ["Embedding models like BERT or sentence-transformers encode text into fixed-size numerical vectors that capture semantic meaning, primarily used for retrieval and classification tasks. Generative models like GPT create new text sequences token-by-token based on learned patterns, used for content creation and conversation."]
        },
        {
            "question": "How does gradient descent optimize neural networks?",
            "answer": "Gradient descent iteratively adjusts model parameters by computing the gradient of the loss function and moving parameters in the opposite direction to minimize error.",
            "contexts": ["Gradient descent is an optimization algorithm that calculates the partial derivatives of the loss function with respect to each parameter, then updates parameters by subtracting a fraction (learning rate) of the gradient. This process repeats until convergence to a local minimum of the loss function."]
        },
        {
            "question": "What is the purpose of batch normalization in deep learning?",
            "answer": "Batch normalization normalizes layer inputs across mini-batches to stabilize and accelerate training by reducing internal covariate shift.",
            "contexts": ["Batch normalization normalizes the inputs of each layer by adjusting and scaling activations using batch statistics (mean and variance). This technique reduces internal covariate shift, allows higher learning rates, reduces sensitivity to initialization, and acts as a form of regularization."]
        },
        {
            "question": "What is the difference between Docker containers and virtual machines?",
            "answer": "Docker containers share the host operating system kernel and virtualize at the application layer, while virtual machines include a full guest OS and virtualize at the hardware layer, making containers more lightweight.",
            "contexts": ["Docker containers package applications with their dependencies but share the host OS kernel, resulting in faster startup times and lower overhead. Virtual machines run complete operating systems on virtualized hardware through a hypervisor, providing stronger isolation but consuming more resources."]
        },
        {
            "question": "What is the purpose of the tokenizer in NLP models?",
            "answer": "A tokenizer breaks down text into smaller units (tokens) that can be processed by the model, converting raw text into numerical representations.",
            "contexts": ["Tokenizers split text into subword units or words and map them to numerical IDs that neural networks can process. Modern tokenizers like BPE (Byte Pair Encoding) and WordPiece balance vocabulary size with representation capability, handling out-of-vocabulary words by breaking them into known subword tokens."]
        },
        {
            "question": "What is mixed precision training in deep learning?",
            "answer": "Mixed precision training uses both 16-bit and 32-bit floating-point types during training to reduce memory usage and increase computational speed while maintaining model accuracy.",
            "contexts": ["Mixed precision training leverages FP16 (16-bit floating point) for most computations to speed up training and reduce memory consumption, while keeping critical operations like loss scaling and parameter updates in FP32 (32-bit) to maintain numerical stability and model convergence."]
        }
    ]

def mock_rag_output_basic():
    return [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "contexts": ["Paris is the capital and largest city of France, located on the Seine River in northern France."]
        },
        {
            "question": "Who wrote Hamlet?",
            "answer": "William Shakespeare",
            "contexts": ["Hamlet is a tragedy written by William Shakespeare between 1599 and 1601."]
        },
        {
            "question": "What is the chemical formula for water?",
            "answer": "The chemical formula for water is H2O.",
            "contexts": ["Water has the chemical formula H2O, meaning each molecule consists of two hydrogen atoms bonded to one oxygen atom."]
        },
        {
            "question": "When did World War II end?",
            "answer": "World War II ended in 1945.",
            "contexts": ["World War II ended in 1945, with Germany surrendering in May and Japan surrendering in September after the atomic bombings."]
        },
        {
            "question": "What is the speed of light?",
            "answer": "The speed of light is approximately 299,792,458 meters per second.",
            "contexts": ["The speed of light in vacuum is exactly 299,792,458 meters per second, often rounded to 3×10^8 m/s."]
        },
        {
            "question": "Who painted the Mona Lisa?",
            "answer": "Leonardo da Vinci painted the Mona Lisa.",
            "contexts": ["The Mona Lisa is a portrait painting by Italian artist Leonardo da Vinci, created between 1503 and 1519."]
        },
        {
            "question": "What is photosynthesis?",
            "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "contexts": ["Photosynthesis is the process used by plants and other organisms to convert light energy into chemical energy stored in glucose, using carbon dioxide and water."]
        },
        {
            "question": "What is the largest planet in our solar system?",
            "answer": "Jupiter is the largest planet in our solar system.",
            "contexts": ["Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined."]
        },
        {
            "question": "Who invented the telephone?",
            "answer": "Alexander Graham Bell is credited with inventing the telephone.",
            "contexts": ["Alexander Graham Bell was awarded the first U.S. patent for the telephone in 1876, though the invention's history involves multiple contributors."]
        },
        {
            "question": "What is the Pythagorean theorem?",
            "answer": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
            "contexts": ["The Pythagorean theorem states that a² + b² = c², where c is the length of the hypotenuse and a and b are the lengths of the other two sides of a right triangle."]
        }
    ]


def create_evaluation_dataset(rag_responses):
    """
    Create evaluation dataset from RAG outputs.
    
    Args:
        rag_responses: List of dicts with 'question', 'answer', 'contexts' keys
    
    Returns:
        EvaluationDataset for ragas evaluation
    """
    samples = []
    for response in rag_responses:
        sample = SingleTurnSample(
            user_input=response["question"],
            response=response["answer"],
            retrieved_contexts=response["contexts"]
        )
        samples.append(sample)
    
    return EvaluationDataset(samples=samples)


@pytest.fixture
def testset():
    data = mock_rag_output_basic()
    return create_evaluation_dataset(data)


@pytest.fixture
def local_llm():
    model_name = "LiquidAI/LFM2-2.6B-GGUF"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7
    )
    
    hf_llm = HuggingFacePipeline(pipeline=pipe)
    return LangchainLLMWrapper(hf_llm)


@pytest.fixture
def local_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    return LangchainEmbeddingsWrapper(embeddings)


def test_faithfulness(testset, local_llm):
    results = evaluate(testset, metrics=[faithfulness], llm=local_llm)
    assert all(0.0 <= r <= 1.0 for r in results["faithfulness"])


def test_answer_relevancy(testset, local_llm, local_embeddings):
    results = evaluate(
        testset, 
        metrics=[answer_relevancy], 
        llm=local_llm, 
        embeddings=local_embeddings
    )
    assert all(0.0 <= r <= 1.0 for r in results["answer_relevancy"])


# Function to use with your actual RAG system
def evaluate_rag_responses(rag_responses, llm, embeddings):
    """
    Evaluate responses from your RAG system.
    
    Args:
        rag_responses: List of dicts with question, answer, contexts
        llm: LangchainLLMWrapper instance
        embeddings: LangchainEmbeddingsWrapper instance
    
    Returns:
        Evaluation results
    """
    dataset = create_evaluation_dataset(rag_responses)
    
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings
    )
    
    return results


if __name__ == "__main__":
    pytest.main([__file__])
