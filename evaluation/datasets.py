"""
Dataset loaders for RAG evaluation.

This module provides a flexible dataset abstraction that makes it easy to:
1. Swap between different datasets (cat-facts, CUAD, custom)
2. Access ground truth for evaluation
3. Chunk documents appropriately per dataset type

Supported datasets:
- cat_facts: Simple text file with one fact per line
- cuad: Contract Understanding Atticus Dataset (legal contracts)
- squad: Stanford Question Answering Dataset
- hotpot_qa: Multi-hop reasoning dataset
- custom: Load any HuggingFace dataset or local files
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

from rag_lite.chunking import TokenChunker

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document with optional metadata."""
    text: str
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = f"doc_{hash(self.text[:100]) % 10000:04d}"


@dataclass
class QAExample:
    """A question-answer example for evaluation."""
    question: str
    answer: str
    context: str  # The ground-truth passage that contains the answer
    example_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.example_id:
            self.example_id = f"qa_{hash(self.question[:50]) % 10000:04d}"


class BaseDataset(ABC):
    """
    Abstract base class for all datasets.
    
    Subclasses must implement:
    - load(): Load the dataset
    - get_documents(): Return documents for indexing
    - get_eval_examples(): Return QA examples for evaluation (if available)
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.documents: List[Document] = []
        self.eval_examples: List[QAExample] = []
        self._loaded = False
        self.config = kwargs
    
    @abstractmethod
    def load(self) -> "BaseDataset":
        """Load the dataset. Returns self for chaining."""
        pass
    
    def get_documents(self) -> List[Document]:
        """Get all documents for indexing."""
        if not self._loaded:
            self.load()
        return self.documents
    
    def get_chunks(self) -> List[str]:
        """Get document texts as a list of strings for indexing."""
        return [doc.text for doc in self.get_documents()]
    
    def get_eval_examples(self) -> List[QAExample]:
        """Get evaluation examples (question-answer pairs with ground truth)."""
        if not self._loaded:
            self.load()
        return self.eval_examples
    
    def has_ground_truth(self) -> bool:
        """Check if this dataset has ground truth for evaluation."""
        return len(self.get_eval_examples()) > 0
    
    def __len__(self) -> int:
        return len(self.get_documents())
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', documents={len(self.documents)}, eval_examples={len(self.eval_examples)})"


class CatFactsDataset(BaseDataset):
    """
    Simple cat facts dataset from a text file.
    One fact per line, no ground truth Q&A.
    """
    
    def __init__(self, file_path: str = "cat-facts.txt", **kwargs):
        super().__init__(name="cat_facts", **kwargs)
        self.file_path = file_path
    
    def load(self) -> "CatFactsDataset":
        """Load cat facts from text file."""
        path = Path(self.file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Cat facts file not found: {self.file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.documents = [
            Document(text=line, doc_id=f"cat_{i:04d}")
            for i, line in enumerate(lines)
        ]
        
        # No ground truth Q&A for this dataset
        self.eval_examples = []
        self._loaded = True
        
        logger.info(f"Loaded {len(self.documents)} cat facts")
        return self


class CUADDataset(BaseDataset):
    """
    Contract Understanding Atticus Dataset (CUAD).
    
    Contains legal contracts with 41 clause types annotated.
    Excellent for testing legal document RAG.
    
    Dataset info:
    - ~500 contracts
    - 13,000+ annotations
    - 41 clause types (e.g., "Governing Law", "Termination", "IP Rights")
    
    Requires: pip install datasets
    """
    
    # CUAD clause types for reference
    CLAUSE_TYPES = [
        "Document Name", "Parties", "Agreement Date", "Effective Date",
        "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
        "Governing Law", "Most Favored Nation", "Non-Compete",
        "Exclusivity", "No-Solicit Of Customers", "Competitive Restriction Exception",
        "No-Solicit Of Employees", "Non-Disparagement", "Termination For Convenience",
        "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
        "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment",
        "Volume Restriction", "Ip Ownership Assignment", "Joint Ip Ownership",
        "License Grant", "Non-Transferable License", "Affiliate License-Licensor",
        "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
        "Irrevocable Or Perpetual License", "Source Code Escrow",
        "Post-Termination Services", "Audit Rights", "Uncapped Liability",
        "Cap On Liability", "Liquidated Damages", "Warranty Duration",
        "Insurance", "Covenant Not To Sue", "Third Party Beneficiary"
    ]
    
    def __init__(
        self,
        split: str = "train",
        max_examples: Optional[int] = None,
        max_eval_examples: int = 100,
        chunk_by_context: bool = True,
        max_tokens: int = 480,  # Safe for 512-token embedding models
        overlap_tokens: int = 80,
        tokenizer_model: str = "BAAI/bge-base-en-v1.5",
        **kwargs
    ):
        """
        Initialize CUAD dataset.
        
        Args:
            split: Dataset split ("train" or "test")
            max_examples: Maximum number of contract contexts to load (None = all)
            max_eval_examples: Maximum evaluation examples to keep
            chunk_by_context: If True, use CUAD's pre-chunked contexts
            max_tokens: Maximum tokens per chunk (default 480 for 512-token models)
            overlap_tokens: Token overlap between chunks for context continuity
            tokenizer_model: HuggingFace tokenizer model name
        """
        super().__init__(name="cuad", **kwargs)
        self.split = split
        self.max_examples = max_examples
        self.max_eval_examples = max_eval_examples
        self.chunk_by_context = chunk_by_context
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer_model = tokenizer_model
        self._chunker = None  # Lazy init
    
    def _get_chunker(self) -> TokenChunker:
        """Get or create the token chunker."""
        if self._chunker is None:
            self._chunker = TokenChunker(
                model_name=self.tokenizer_model,
                max_tokens=self.max_tokens,
                overlap_tokens=self.overlap_tokens,
            )
        return self._chunker
    
    def load(self) -> "CUADDataset":
        """Load CUAD dataset from HuggingFace."""
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets required. Install with: pip install datasets"
            )
        
        logger.info(f"Loading CUAD dataset (split={self.split})...")
        
        # Load from HuggingFace
        # Try multiple approaches since dataset format varies by HF datasets version
        dataset = None
        errors = []
        
        # Approach 1: Use auto-converted parquet branch (works with HF datasets 4.x+)
        try:
            dataset = hf_load_dataset(
                "theatticusproject/cuad-qa",
                revision="refs/convert/parquet",
                split=self.split,
            )
            logger.info("Loaded CUAD from theatticusproject/cuad-qa (parquet branch)")
        except Exception as e:
            errors.append(f"cuad-qa parquet branch: {e}")
        
        # Approach 2: Try loading with older HF datasets that support scripts
        if dataset is None:
            try:
                # This works with older datasets versions
                import datasets
                if hasattr(datasets, '__version__') and datasets.__version__ < "4.0.0":
                    dataset = hf_load_dataset("theatticusproject/cuad-qa", split=self.split, trust_remote_code=True)
                    logger.info("Loaded CUAD from theatticusproject/cuad-qa (legacy)")
            except Exception as e:
                errors.append(f"cuad-qa legacy: {e}")
        
        # Approach 3: Use SQuAD as fallback (always works, similar format)
        if dataset is None:
            logger.warning(f"Could not load CUAD dataset. Errors: {errors}")
            logger.warning("Falling back to SQuAD dataset (similar QA format)")
            dataset = hf_load_dataset("squad", split=self.split)
            self.name = "squad_fallback"
        
        # Track unique contexts and their questions
        context_to_doc: Dict[str, Document] = {}
        context_to_questions: Dict[str, List[QAExample]] = {}
        
        for i, item in enumerate(dataset):
            context = item['context'].strip()
            question = item['question']
            answers = item['answers']
            title = item.get('title', '')
            
            # Skip if we've reached max examples
            if self.max_examples and len(context_to_doc) >= self.max_examples:
                # But still collect Q&A for existing contexts
                if context not in context_to_doc:
                    continue
            
            # Create document if new context
            if context not in context_to_doc:
                doc_id = f"cuad_{len(context_to_doc):04d}"
                context_to_doc[context] = Document(
                    text=context,
                    doc_id=doc_id,
                    metadata={"title": title, "source": "cuad"}
                )
                context_to_questions[context] = []
            
            # Create Q&A example if there's an answer
            if answers['text'] and len(answers['text']) > 0:
                answer_text = answers['text'][0]
                qa = QAExample(
                    question=question,
                    answer=answer_text,
                    context=context,
                    example_id=f"cuad_qa_{i:05d}",
                    metadata={
                        "title": title,
                        "answer_start": answers['answer_start'][0] if answers['answer_start'] else -1
                    }
                )
                context_to_questions[context].append(qa)
        
        # Store documents - chunk using token-aware chunking for embedding model compatibility
        raw_docs = list(context_to_doc.values())
        self.documents = []
        
        # Initialize token chunker
        chunker = self._get_chunker()
        
        for doc in raw_docs:
            # Use token count to decide if chunking is needed
            if chunker.token_count(doc.text) <= self.max_tokens:
                self.documents.append(doc)
            else:
                # Chunk long documents using token-aware chunker
                chunks = chunker.chunk_text(doc.text)
                for idx, chunk_text in enumerate(chunks):
                    self.documents.append(Document(
                        text=chunk_text,
                        doc_id=f"{doc.doc_id}_chunk{idx:02d}",
                        metadata={**doc.metadata, "parent_id": doc.doc_id, "chunk_idx": idx}
                    ))
        
        logger.info(f"Chunked {len(raw_docs)} contexts into {len(self.documents)} chunks (token-based)")
        
        # Collect all Q&A examples (limit to max_eval_examples)
        all_examples = []
        for context, examples in context_to_questions.items():
            all_examples.extend(examples)
        
        # Sample diverse examples if we have too many
        if len(all_examples) > self.max_eval_examples:
            import random
            random.seed(42)  # Reproducibility
            self.eval_examples = random.sample(all_examples, self.max_eval_examples)
        else:
            self.eval_examples = all_examples
        
        self._loaded = True
        
        logger.info(
            f"Loaded CUAD: {len(self.documents)} chunks from {len(raw_docs)} documents, "
            f"{len(self.eval_examples)} eval examples"
        )
        return self
    
    def _chunk_text(self, text: str, max_chars: int, overlap: int) -> List[str]:
        """
        Split text into chunks with overlap, trying to break at sentence boundaries.
        
        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            
            # Try to find a sentence boundary (., !, ?) near the end
            # Look back up to 200 chars for a good break point
            best_break = end
            for i in range(end, max(start + max_chars - 200, start), -1):
                if text[i] in '.!?\n':
                    best_break = i + 1
                    break
            
            chunk = text[start:best_break].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = best_break - overlap
            if start < 0:
                start = 0
        
        return chunks
    
    def get_clause_questions(self, clause_type: str) -> List[QAExample]:
        """Get evaluation examples for a specific clause type."""
        clause_lower = clause_type.lower()
        return [
            ex for ex in self.eval_examples
            if clause_lower in ex.question.lower()
        ]


class HuggingFaceDataset(BaseDataset):
    """
    Generic loader for any HuggingFace dataset.
    
    Supports datasets with various structures by specifying field mappings.
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_field: str = "context",
        question_field: Optional[str] = "question",
        answer_field: Optional[str] = "answers",
        max_examples: Optional[int] = None,
        subset: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize HuggingFace dataset loader.
        
        Args:
            dataset_name: Name of the dataset (e.g., "squad", "hotpot_qa")
            split: Dataset split
            text_field: Field name for document text
            question_field: Field name for questions (None if no Q&A)
            answer_field: Field name for answers (None if no Q&A)
            max_examples: Maximum examples to load
            subset: Dataset subset/config name (e.g., "fullwiki" for hotpot_qa)
        """
        super().__init__(name=dataset_name, **kwargs)
        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.question_field = question_field
        self.answer_field = answer_field
        self.max_examples = max_examples
        self.subset = subset
    
    def load(self) -> "HuggingFaceDataset":
        """Load dataset from HuggingFace."""
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets required. Install with: pip install datasets"
            )
        
        logger.info(f"Loading {self.dataset_name} from HuggingFace...")
        
        # Load dataset
        if self.subset:
            dataset = hf_load_dataset(self.dataset_name, self.subset, split=self.split, trust_remote_code=True)
        else:
            dataset = hf_load_dataset(self.dataset_name, split=self.split, trust_remote_code=True)
        
        # Limit if specified
        if self.max_examples:
            dataset = dataset.select(range(min(self.max_examples, len(dataset))))
        
        # Extract documents and Q&A
        seen_texts = set()
        
        for i, item in enumerate(dataset):
            text = item.get(self.text_field, "")
            if not text or text in seen_texts:
                continue
            
            seen_texts.add(text)
            
            # Create document
            doc = Document(
                text=text,
                doc_id=f"{self.name}_{len(self.documents):05d}",
                metadata={"source": self.dataset_name}
            )
            self.documents.append(doc)
            
            # Create Q&A example if fields are specified
            if self.question_field and self.answer_field:
                question = item.get(self.question_field, "")
                answers = item.get(self.answer_field, {})
                
                # Handle different answer formats
                if isinstance(answers, dict):
                    answer_text = answers.get('text', [''])[0] if answers.get('text') else ""
                elif isinstance(answers, list):
                    answer_text = answers[0] if answers else ""
                else:
                    answer_text = str(answers)
                
                if question and answer_text:
                    qa = QAExample(
                        question=question,
                        answer=answer_text,
                        context=text,
                        example_id=f"{self.name}_qa_{i:05d}"
                    )
                    self.eval_examples.append(qa)
        
        self._loaded = True
        logger.info(
            f"Loaded {self.dataset_name}: {len(self.documents)} documents, "
            f"{len(self.eval_examples)} eval examples"
        )
        return self


class CustomDataset(BaseDataset):
    """
    Custom dataset from local files or provided data.
    
    Supports:
    - Text files (one document per line or whole file)
    - JSON/JSONL files with documents and optional Q&A
    - Direct data injection
    """
    
    def __init__(
        self,
        name: str = "custom",
        file_path: Optional[str] = None,
        documents: Optional[List[str]] = None,
        eval_examples: Optional[List[Dict]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 50,
        **kwargs
    ):
        """
        Initialize custom dataset.
        
        Args:
            name: Dataset name
            file_path: Path to data file (.txt, .json, .jsonl)
            documents: List of document texts (alternative to file)
            eval_examples: List of Q&A dicts with keys: question, answer, context
            chunk_size: If set, chunk documents into smaller pieces
            chunk_overlap: Overlap between chunks (characters)
        """
        super().__init__(name=name, **kwargs)
        self.file_path = file_path
        self._provided_documents = documents
        self._provided_examples = eval_examples
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load(self) -> "CustomDataset":
        """Load custom dataset."""
        # Load from provided data
        if self._provided_documents:
            self.documents = [
                Document(text=text, doc_id=f"custom_{i:04d}")
                for i, text in enumerate(self._provided_documents)
            ]
        
        # Or load from file
        elif self.file_path:
            path = Path(self.file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")
            
            if path.suffix == '.json':
                self._load_json(path)
            elif path.suffix == '.jsonl':
                self._load_jsonl(path)
            else:
                self._load_text(path)
        
        # Load eval examples
        if self._provided_examples:
            self.eval_examples = [
                QAExample(
                    question=ex['question'],
                    answer=ex['answer'],
                    context=ex.get('context', ''),
                    example_id=f"custom_qa_{i:04d}"
                )
                for i, ex in enumerate(self._provided_examples)
            ]
        
        # Optional chunking
        if self.chunk_size:
            self._chunk_documents()
        
        self._loaded = True
        logger.info(f"Loaded custom dataset: {len(self.documents)} documents")
        return self
    
    def _load_text(self, path: Path):
        """Load from text file."""
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.documents = [
            Document(text=line, doc_id=f"custom_{i:04d}")
            for i, line in enumerate(lines)
        ]
    
    def _load_json(self, path: Path):
        """Load from JSON file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    self.documents.append(Document(text=item, doc_id=f"custom_{i:04d}"))
                elif isinstance(item, dict):
                    text = item.get('text') or item.get('content') or item.get('document', '')
                    if text:
                        self.documents.append(Document(
                            text=text,
                            doc_id=item.get('id', f"custom_{i:04d}"),
                            metadata=item.get('metadata', {})
                        ))
    
    def _load_jsonl(self, path: Path):
        """Load from JSONL file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    item = json.loads(line)
                    text = item.get('text') or item.get('content') or item.get('document', '')
                    if text:
                        self.documents.append(Document(
                            text=text,
                            doc_id=item.get('id', f"custom_{i:04d}"),
                            metadata=item.get('metadata', {})
                        ))
    
    def _chunk_documents(self):
        """Chunk documents into smaller pieces."""
        chunked = []
        for doc in self.documents:
            text = doc.text
            if len(text) <= self.chunk_size:
                chunked.append(doc)
            else:
                # Chunk with overlap
                start = 0
                chunk_idx = 0
                while start < len(text):
                    end = start + self.chunk_size
                    chunk_text = text[start:end]
                    
                    chunked.append(Document(
                        text=chunk_text,
                        doc_id=f"{doc.doc_id}_chunk{chunk_idx:02d}",
                        metadata={**doc.metadata, "parent_id": doc.doc_id}
                    ))
                    
                    start = end - self.chunk_overlap
                    chunk_idx += 1
        
        self.documents = chunked


# Dataset registry for easy access
DATASET_REGISTRY = {
    "cat_facts": CatFactsDataset,
    "cuad": CUADDataset,
    "squad": lambda **kwargs: HuggingFaceDataset(
        dataset_name="squad",
        text_field="context",
        question_field="question",
        answer_field="answers",
        **kwargs
    ),
    "hotpot_qa": lambda **kwargs: HuggingFaceDataset(
        dataset_name="hotpot_qa",
        subset="fullwiki",
        text_field="context",
        question_field="question",
        answer_field="answer",
        **kwargs
    ),
    "natural_questions": lambda **kwargs: HuggingFaceDataset(
        dataset_name="google-research-datasets/natural_questions",
        subset="default",
        **kwargs
    ),
}


def load_dataset(
    name: str,
    **kwargs
) -> BaseDataset:
    """
    Load a dataset by name.
    
    Args:
        name: Dataset name (cat_facts, cuad, squad, hotpot_qa, or custom)
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        Loaded dataset instance
        
    Example:
        >>> dataset = load_dataset("cuad", max_examples=100)
        >>> chunks = dataset.get_chunks()
        >>> eval_examples = dataset.get_eval_examples()
    """
    if name in DATASET_REGISTRY:
        dataset_cls = DATASET_REGISTRY[name]
        if callable(dataset_cls) and not isinstance(dataset_cls, type):
            # It's a factory function
            dataset = dataset_cls(**kwargs)
        else:
            # It's a class
            dataset = dataset_cls(**kwargs)
        return dataset.load()
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(DATASET_REGISTRY.keys())} or use CustomDataset"
        )


def list_available_datasets() -> List[str]:
    """List all available dataset names."""
    return list(DATASET_REGISTRY.keys())
