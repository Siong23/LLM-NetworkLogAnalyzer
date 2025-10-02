# analysis_rag_index.py - CORRECTED VERSION WITH ALL METHODS
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class AnalysisRAGIndex:
    """
    Enhanced FAISS-based RAG for completed LLM analyses.
    Stores (embedding, text, metadata) and provides sophisticated search capabilities.
    """

    def __init__(
        self, 
        index_dir: str = "faiss_indices/analysis_rag", 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
        device: str = "cpu"
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.device = device

        self.model = None
        self.index = None
        self.dim = None
        self.text_store: List[str] = []
        self.meta_store: List[Dict] = []

        # Enhanced search features
        self.attack_type_index = {}  # Maps attack types to document indices
        self.timestamp_index = []    # Sorted list of (timestamp, doc_index) pairs
        
        self._load()

    def _ensure_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            if SentenceTransformer is None:
                raise RuntimeError("Please install sentence-transformers to use AnalysisRAGIndex.")
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded on device: {self.device}")

    def _load(self):
        """Load existing index and metadata from disk."""
        meta_path = self.index_dir / "store.json"
        idx_path = self.index_dir / "index.faiss"

        if meta_path.exists() and idx_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    blob = json.load(f)
                self.text_store = blob.get("texts", [])
                self.meta_store = blob.get("metas", [])
                self.dim = blob.get("dim", None)
                self.model_name = blob.get("model_name", self.model_name)

                self.index = faiss.read_index(str(idx_path))
                
                # Rebuild auxiliary indices
                self._rebuild_auxiliary_indices()
                
                logger.info(f"Loaded RAG index with {len(self.text_store)} documents")
            except Exception as e:
                logger.error(f"Error loading RAG index: {e}")
                self._initialize_empty()
        else:
            self._initialize_empty()

    def _initialize_empty(self):
        """Initialize empty RAG index."""
        self.text_store = []
        self.meta_store = []
        self.index = None
        self.dim = None
        self.attack_type_index = {}
        self.timestamp_index = []

    def _rebuild_auxiliary_indices(self):
        """Rebuild auxiliary indices for faster searching."""
        self.attack_type_index = {}
        self.timestamp_index = []
        
        for i, meta in enumerate(self.meta_store):
            # Attack type index
            attack_type = meta.get("attack_type", "unknown").lower()
            if attack_type not in self.attack_type_index:
                self.attack_type_index[attack_type] = []
            self.attack_type_index[attack_type].append(i)
            
            # Timestamp index
            timestamp = meta.get("timestamp", 0)
            self.timestamp_index.append((timestamp, i))
        
        # Sort timestamp index
        self.timestamp_index.sort(key=lambda x: x[0], reverse=True)  # Most recent first
        
        logger.info(f"Rebuilt auxiliary indices: {len(self.attack_type_index)} attack types")

    def _save(self):
        """Save index and metadata to disk."""
        try:
            meta_path = self.index_dir / "store.json"
            idx_path = self.index_dir / "index.faiss"
            
            blob = {
                "texts": self.text_store,
                "metas": self.meta_store,
                "dim": self.dim,
                "model_name": self.model_name,
                "created_at": datetime.now().isoformat(),
                "total_documents": len(self.text_store)
            }
            
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
            
            if self.index is not None:
                faiss.write_index(self.index, str(idx_path))
                
            logger.debug(f"Saved RAG index with {len(self.text_store)} documents")
        except Exception as e:
            logger.error(f"Error saving RAG index: {e}")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the sentence transformer model."""
        self._ensure_model()
        try:
            vecs = self.model.encode(
                texts, 
                convert_to_numpy=True, 
                show_progress_bar=False, 
                normalize_embeddings=True,
                batch_size=32
            )
            return vecs.astype(np.float32)
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise

    def add_document(self, text: str, metadata: Dict) -> int:
        """
        Add a document to the RAG index.
        
        Args:
            text: The text content to index
            metadata: Dictionary containing metadata about the document
            
        Returns:
            Document index (integer ID)
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.now().timestamp()
            
            # Embed the text
            vec = self._embed([text])
            
            # Initialize index if needed
            if self.index is None:
                self.dim = int(vec.shape[1])
                self.index = faiss.IndexFlatIP(self.dim)  # Inner product with normalized vecs == cosine
                logger.info(f"Initialized FAISS index with dimension {self.dim}")
            
            # Add to FAISS index
            self.index.add(vec)
            
            # Add to stores
            doc_index = len(self.text_store)
            self.text_store.append(text)
            self.meta_store.append(metadata)
            
            # Update auxiliary indices
            attack_type = metadata.get("attack_type", "unknown").lower()
            if attack_type not in self.attack_type_index:
                self.attack_type_index[attack_type] = []
            self.attack_type_index[attack_type].append(doc_index)
            
            timestamp = metadata.get("timestamp", 0)
            self.timestamp_index.append((timestamp, doc_index))
            self.timestamp_index.sort(key=lambda x: x[0], reverse=True)
            
            # Save to disk
            self._save()
            
            logger.info(f"Added document {doc_index} for attack type: {attack_type}")
            return doc_index
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise

    def search(self, query: str, k: int = 5, attack_type_filter: str = None, recent_only: bool = False) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            attack_type_filter: Optional filter by attack type
            recent_only: If True, only search recent documents (last 100)
            
        Returns:
            List of dictionaries with text, metadata, and score
        """
        if self.index is None or len(self.text_store) == 0:
            return []
        
        try:
            # Embed query
            q = self._embed([query])
            
            # Determine search space
            search_indices = None
            
            if attack_type_filter:
                # Filter by attack type
                attack_type_filter = attack_type_filter.lower()
                search_indices = self.attack_type_index.get(attack_type_filter, [])
                if not search_indices:
                    logger.warning(f"No documents found for attack type: {attack_type_filter}")
                    return []
            elif recent_only and len(self.timestamp_index) > 100:
                # Use only recent documents
                search_indices = [idx for _, idx in self.timestamp_index[:100]]
            
            if search_indices is not None:
                # Create subset index for filtered search
                if not search_indices:
                    return []
                
                # Get vectors for subset
                subset_vectors = np.array([
                    self.index.reconstruct(idx) for idx in search_indices
                ]).astype(np.float32)
                
                # Create temporary index
                temp_index = faiss.IndexFlatIP(self.dim)
                temp_index.add(subset_vectors)
                
                # Search in subset
                sims, idxs = temp_index.search(q, min(k, len(search_indices)))
                
                # Map back to original indices
                results = []
                for i, sim_idx in enumerate(idxs[0]):
                    if 0 <= sim_idx < len(search_indices):
                        original_idx = search_indices[sim_idx]
                        score = float(sims[0][i])
                        results.append({
                            "text": self.text_store[original_idx],
                            "metadata": self.meta_store[original_idx],
                            "score": score,
                            "doc_index": original_idx
                        })
                return results
            else:
                # Search entire index
                sims, idxs = self.index.search(q, min(k, len(self.text_store)))
                
                results = []
                for i, idx in enumerate(idxs[0]):
                    if 0 <= idx < len(self.text_store):
                        score = float(sims[0][i])
                        results.append({
                            "text": self.text_store[idx],
                            "metadata": self.meta_store[idx],
                            "score": score,
                            "doc_index": idx
                        })
                return results
                
        except Exception as e:
            logger.error(f"Error searching RAG index: {e}")
            return []

    def search_by_attack_type(self, attack_type: str, k: int = 5) -> List[Dict]:
        """Search for documents of a specific attack type."""
        return self.search("", k=k, attack_type_filter=attack_type)

    def get_recent_analyses(self, k: int = 10) -> List[Dict]:
        """Get the most recent analyses."""
        results = []
        for i, (timestamp, doc_idx) in enumerate(self.timestamp_index[:k]):
            if doc_idx < len(self.text_store):
                results.append({
                    "text": self.text_store[doc_idx],
                    "metadata": self.meta_store[doc_idx],
                    "score": 1.0,  # No similarity score for recent search
                    "doc_index": doc_idx
                })
        return results

    def get_stats(self) -> Dict:
        """Get statistics about the RAG index."""
        stats = {
            "total_documents": len(self.text_store),
            "attack_types": list(self.attack_type_index.keys()),
            "attack_type_counts": {
                attack_type: len(indices) 
                for attack_type, indices in self.attack_type_index.items()
            },
            "index_dimension": self.dim,
            "model_name": self.model_name,
            "device": self.device
        }
        
        if self.timestamp_index:
            recent_timestamp = self.timestamp_index[0][0]
            oldest_timestamp = self.timestamp_index[-1][0]
            stats.update({
                "most_recent": datetime.fromtimestamp(recent_timestamp).isoformat(),
                "oldest": datetime.fromtimestamp(oldest_timestamp).isoformat()
            })
        
        return stats

    def remove_old_documents(self, keep_recent: int = 1000):
        """
        Remove old documents to keep index size manageable.
        
        Args:
            keep_recent: Number of recent documents to keep
        """
        if len(self.text_store) <= keep_recent:
            return
        
        logger.info(f"Removing old documents, keeping {keep_recent} most recent")
        
        # Get indices of documents to keep (most recent)
        keep_indices = set()
        for i, (_, doc_idx) in enumerate(self.timestamp_index[:keep_recent]):
            keep_indices.add(doc_idx)
        
        # Create new stores with only kept documents
        new_text_store = []
        new_meta_store = []
        index_mapping = {}  # old_index -> new_index
        
        for old_idx in sorted(keep_indices):
            new_idx = len(new_text_store)
            new_text_store.append(self.text_store[old_idx])
            new_meta_store.append(self.meta_store[old_idx])
            index_mapping[old_idx] = new_idx
        
        # Rebuild FAISS index
        if new_text_store:
            vectors = []
            for old_idx in sorted(keep_indices):
                vectors.append(self.index.reconstruct(old_idx))
            
            vectors = np.array(vectors).astype(np.float32)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(vectors)
        else:
            self.index = None
        
        # Update stores
        self.text_store = new_text_store
        self.meta_store = new_meta_store
        
        # Rebuild auxiliary indices
        self._rebuild_auxiliary_indices()
        
        # Save
        self._save()
        
        logger.info(f"Removed {len(self.text_store) - len(new_text_store)} old documents")

    def clear_index(self):
        """Clear all documents from the index."""
        logger.warning("Clearing entire RAG index")
        self._initialize_empty()
        self._save()

    def export_documents(self, output_path: str):
        """Export all documents to a JSON file."""
        export_data = {
            "metadata": {
                "total_documents": len(self.text_store),
                "model_name": self.model_name,
                "exported_at": datetime.now().isoformat()
            },
            "documents": [
                {
                    "index": i,
                    "text": text,
                    "metadata": meta
                }
                for i, (text, meta) in enumerate(zip(self.text_store, self.meta_store))
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(self.text_store)} documents to {output_path}")

    def __len__(self):
        """Return number of documents in the index."""
        return len(self.text_store)

    def __contains__(self, doc_index: int):
        """Check if document index exists."""
        return 0 <= doc_index < len(self.text_store)


def main():
    """Example usage and testing."""
    logging.basicConfig(level=logging.INFO)
    
    # Create RAG index
    rag = AnalysisRAGIndex()
    
    # Add some example documents
    example_docs = [
        {
            "text": "SYN flood attack detected. High volume of SYN packets without corresponding ACK responses. Recommend blocking source IP and implementing SYN cookies.",
            "metadata": {
                "attack_type": "synflood",
                "threat_level": "HIGH",
                "risk_score": 85,
                "protocol": "TCP"
            }
        },
        {
            "text": "ICMP flood attack identified. Excessive ICMP echo requests causing network congestion. Immediate action: rate limit ICMP traffic.",
            "metadata": {
                "attack_type": "icmpflood", 
                "threat_level": "MEDIUM",
                "risk_score": 70,
                "protocol": "ICMP"
            }
        }
    ]
    
    for doc in example_docs:
        rag.add_document(doc["text"], doc["metadata"])
    
    # Test searches
    print("\n=== RAG Index Statistics ===")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Search Results ===")
    results = rag.search("SYN attack network flood", k=2)
    for i, result in enumerate(results):
        print(f"Result {i+1} (score: {result['score']:.3f}):")
        print(f"  Attack Type: {result['metadata']['attack_type']}")
        print(f"  Text: {result['text'][:100]}...")
        print()


if __name__ == "__main__":
    main()