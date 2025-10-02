#!/usr/bin/env python3
"""
RAG Management Utility for Network IDS

This script provides utilities to manage, inspect, and maintain the RAG (Retrieval-Augmented Generation) 
index used by the Network Intrusion Detection System.
"""

import argparse
import logging
from pathlib import Path
import json
import sys
from datetime import datetime
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from analysis_rag_index import AnalysisRAGIndex
except ImportError:
    logger.error("Could not import AnalysisRAGIndex. Make sure analysis_rag_index.py is available.")
    sys.exit(1)


class RAGManager:
    """Utility class for managing the RAG index."""
    
    def __init__(self, index_dir: str = "faiss_indices/analysis_rag"):
        self.index_dir = index_dir
        self.rag = None
    
    def _ensure_rag(self):
        """Ensure RAG index is loaded."""
        if self.rag is None:
            try:
                self.rag = AnalysisRAGIndex(index_dir=self.index_dir)
                logger.info(f"Loaded RAG index from {self.index_dir}")
            except Exception as e:
                logger.error(f"Failed to load RAG index: {e}")
                raise
    
    def show_stats(self):
        """Display comprehensive statistics about the RAG index."""
        self._ensure_rag()
        
        stats = self.rag.get_stats()
        
        print("=" * 60)
        print("📚 RAG INDEX STATISTICS")
        print("=" * 60)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Index Dimension: {stats['index_dimension']}")
        print(f"Model: {stats['model_name']}")
        print(f"Device: {stats['device']}")
        
        if 'most_recent' in stats:
            print(f"Most Recent: {stats['most_recent']}")
            print(f"Oldest: {stats['oldest']}")
        
        print(f"\nAttack Types: {len(stats['attack_types'])}")
        for attack_type in sorted(stats['attack_types']):
            count = stats['attack_type_counts'][attack_type]
            print(f"  {attack_type}: {count} documents")
        
        print("=" * 60)
    
    def search_interactive(self):
        """Interactive search interface."""
        self._ensure_rag()
        
        print("=" * 60)
        print("🔍 INTERACTIVE RAG SEARCH")
        print("=" * 60)
        print("Enter search queries (or 'quit' to exit)")
        print("You can also use:")
        print("  'recent' - Show recent analyses")
        print("  'stats' - Show statistics")
        print("  'types' - List attack types")
        print()
        
        while True:
            try:
                query = input("Search> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'stats':
                    self.show_stats()
                    continue
                elif query.lower() == 'recent':
                    self._show_recent()
                    continue
                elif query.lower() == 'types':
                    self._show_attack_types()
                    continue
                elif not query:
                    continue
                
                # Perform search
                results = self.rag.search(query, k=5)
                
                if not results:
                    print("No results found.")
                    continue
                
                print(f"\nFound {len(results)} results:")
                print("-" * 40)
                
                for i, result in enumerate(results):
                    meta = result['metadata']
                    score = result['score']
                    
                    print(f"Result {i+1} (Score: {score:.3f})")
                    print(f"  Attack Type: {meta.get('attack_type', 'Unknown')}")
                    print(f"  Threat Level: {meta.get('threat_level', 'Unknown')}")
                    print(f"  Risk Score: {meta.get('risk_score', 'Unknown')}")
                    
                    if 'timestamp' in meta:
                        timestamp = datetime.fromtimestamp(meta['timestamp'])
                        print(f"  Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Show truncated text
                    text = result['text']
                    if len(text) > 200:
                        text = text[:200] + "..."
                    print(f"  Text: {text}")
                    print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_recent(self):
        """Show recent analyses."""
        recent = self.rag.get_recent_analyses(k=10)
        
        if not recent:
            print("No recent analyses found.")
            return
        
        print(f"\n📅 {len(recent)} Most Recent Analyses:")
        print("-" * 40)
        
        for i, doc in enumerate(recent):
            meta = doc['metadata']
            timestamp = datetime.fromtimestamp(meta.get('timestamp', 0))
            
            print(f"{i+1}. {meta.get('attack_type', 'Unknown')} - {meta.get('threat_level', 'Unknown')}")
            print(f"   {timestamp.strftime('%Y-%m-%d %H:%M:%S')} (Risk: {meta.get('risk_score', 'Unknown')})")
            
            # Show brief text
            text = doc['text']
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"   {text}")
            print()
    
    def _show_attack_types(self):
        """Show available attack types."""
        stats = self.rag.get_stats()
        
        print(f"\n🚨 Available Attack Types ({len(stats['attack_types'])}):")
        print("-" * 40)
        
        for attack_type in sorted(stats['attack_types']):
            count = stats['attack_type_counts'][attack_type]
            print(f"  {attack_type}: {count} documents")
    
    def search_by_type(self, attack_type: str, k: int = 5):
        """Search for documents of a specific attack type."""
        self._ensure_rag()
        
        results = self.rag.search_by_attack_type(attack_type, k=k)
        
        if not results:
            print(f"No documents found for attack type: {attack_type}")
            return
        
        print(f"📋 {len(results)} Documents for Attack Type: {attack_type}")
        print("-" * 50)
        
        for i, result in enumerate(results):
            meta = result['metadata']
            timestamp = datetime.fromtimestamp(meta.get('timestamp', 0))
            
            print(f"Document {i+1}:")
            print(f"  Threat Level: {meta.get('threat_level', 'Unknown')}")
            print(f"  Risk Score: {meta.get('risk_score', 'Unknown')}")
            print(f"  Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            text = result['text']
            if len(text) > 300:
                text = text[:300] + "..."
            print(f"  Text: {text}")
            print()
    
    def export_data(self, output_path: str):
        """Export RAG data to JSON file."""
        self._ensure_rag()
        
        try:
            self.rag.export_documents(output_path)
            print(f"✅ Exported RAG data to: {output_path}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def cleanup_old(self, keep_recent: int = 1000):
        """Remove old documents from the index."""
        self._ensure_rag()
        
        old_count = len(self.rag)
        
        if old_count <= keep_recent:
            print(f"Index has {old_count} documents, no cleanup needed (keeping {keep_recent})")
            return
        
        print(f"Cleaning up old documents (keeping {keep_recent} most recent)...")
        
        try:
            self.rag.remove_old_documents(keep_recent=keep_recent)
            new_count = len(self.rag)
            removed = old_count - new_count
            
            print(f"✅ Cleanup complete:")
            print(f"  Removed: {removed} documents")
            print(f"  Remaining: {new_count} documents")
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
    
    def clear_index(self, confirm: bool = False):
        """Clear the entire RAG index."""
        self._ensure_rag()
        
        if not confirm:
            print("⚠️  This will permanently delete all RAG data!")
            response = input("Type 'YES' to confirm: ").strip()
            if response != 'YES':
                print("Cancelled.")
                return
        
        try:
            self.rag.clear_index()
            print("✅ RAG index cleared successfully")
        except Exception as e:
            print(f"❌ Clear failed: {e}")
    
    def test_search(self, queries: List[str]):
        """Test search with predefined queries."""
        self._ensure_rag()
        
        print("🧪 TESTING RAG SEARCH")
        print("=" * 50)
        
        for i, query in enumerate(queries):
            print(f"\nTest {i+1}: '{query}'")
            print("-" * 30)
            
            results = self.rag.search(query, k=3)
            
            if not results:
                print("No results found.")
                continue
            
            for j, result in enumerate(results):
                meta = result['metadata']
                score = result['score']
                
                print(f"  {j+1}. {meta.get('attack_type', 'Unknown')} (Score: {score:.3f})")
                print(f"     Threat: {meta.get('threat_level', 'Unknown')}, Risk: {meta.get('risk_score', 'Unknown')}")


def main():
    parser = argparse.ArgumentParser(description="RAG Management Utility for Network IDS")
    parser.add_argument("--index-dir", default="faiss_indices/analysis_rag",
                       help="Path to RAG index directory")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    subparsers.add_parser('stats', help='Show RAG index statistics')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Interactive search interface')
    
    # Search by type command
    type_parser = subparsers.add_parser('search-type', help='Search by attack type')
    type_parser.add_argument('attack_type', help='Attack type to search for')
    type_parser.add_argument('-k', '--count', type=int, default=5, help='Number of results')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export RAG data to JSON')
    export_parser.add_argument('output', help='Output file path')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Remove old documents')
    cleanup_parser.add_argument('-k', '--keep', type=int, default=1000, 
                               help='Number of recent documents to keep')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear entire RAG index')
    clear_parser.add_argument('--force', action='store_true', 
                             help='Skip confirmation prompt')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test search functionality')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize manager
    manager = RAGManager(index_dir=args.index_dir)
    
    try:
        if args.command == 'stats':
            manager.show_stats()
            
        elif args.command == 'search':
            manager.search_interactive()
            
        elif args.command == 'search-type':
            manager.search_by_type(args.attack_type, args.count)
            
        elif args.command == 'export':
            manager.export_data(args.output)
            
        elif args.command == 'cleanup':
            manager.cleanup_old(args.keep)
            
        elif args.command == 'clear':
            manager.clear_index(args.force)
            
        elif args.command == 'test':
            test_queries = [
                "SYN flood attack high volume packets",
                "ICMP flooding network congestion",
                "HTTP flood application layer attack",
                "port scanning reconnaissance",
                "UDP flood denial of service"
            ]
            manager.test_search(test_queries)
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())