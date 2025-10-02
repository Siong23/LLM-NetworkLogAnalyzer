#!/usr/bin/env python3
"""
Fixed setup script for the Network Intrusion Detection System.
Handles subfolder data structures and provides better diagnostics.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedNetworkIDSSetup:
    """Fixed setup manager for the Network IDS that handles subfolder structures."""
    
    def __init__(self):
        self.data_dir = Path("data_split")
        self.index_dir = Path("faiss_indices")
    
    def inspect_data_structure(self):
        """Inspect the data directory structure."""
        logger.info("=== Inspecting Data Structure ===")
        
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return False
        
        # Find all CSV files
        csv_files = list(self.data_dir.rglob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in total")
        
        if not csv_files:
            logger.error("No CSV files found!")
            logger.error("Please ensure your data is properly extracted to the data_split directory")
            return False
        
        # Categorize files
        benign_files = []
        attack_files = []
        
        for csv_file in csv_files:
            filename = csv_file.stem.lower()
            rel_path = csv_file.relative_to(self.data_dir)
            
            if 'benign' in filename:
                benign_files.append(rel_path)
            elif any(attack in filename for attack in ['icmp', 'udp', 'syn', 'http', 'scan', 'dos', 'flood']):
                attack_files.append(rel_path)
            else:
                logger.info(f"Unknown file type: {rel_path}")
        
        logger.info(f"Found {len(benign_files)} benign files:")
        for f in benign_files:
            logger.info(f"  - {f}")
        
        logger.info(f"Found {len(attack_files)} attack files:")
        for f in attack_files:
            logger.info(f"  - {f}")
        
        if len(benign_files) == 0:
            logger.warning("No benign files detected! Look for files with 'benign' in the name")
        
        if len(attack_files) < 3:
            logger.warning("Very few attack files detected! Expected multiple attack types")
        
        return len(csv_files) > 0
    
    def check_requirements(self):
        """Check if all requirements are met."""
        logger.info("=== Checking System Requirements ===")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required. Current version: {}.{}.{}".format(*sys.version_info[:3]))
            return False
        logger.info(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"✓ GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                logger.warning("⚠ No GPU detected. CPU fallback will be used (much slower)")
        except ImportError:
            logger.warning("⚠ PyTorch not installed. GPU acceleration will not be available")
        
        # Check data structure
        if not self.inspect_data_structure():
            return False
        
        logger.info("✓ Data structure looks good")
        return True
    
    def install_dependencies(self):
        """Install required dependencies."""
        logger.info("=== Installing Dependencies ===")
        
        try:
            # Check if requirements.txt exists
            if not Path("requirements.txt").exists():
                logger.error("requirements.txt not found!")
                return False
            
            # Install dependencies
            logger.info("Installing Python packages...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False
            
            logger.info("✓ Dependencies installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
    
    def build_faiss_indices(self):
        """Build FAISS indices from the dataset."""
        logger.info("=== Building FAISS Indices ===")
        
        try:
            # Check if indices already exist
            if self.index_dir.exists() and (self.index_dir / "preprocessing.pkl").exists():
                logger.info("FAISS indices already exist.")
                response = input("Do you want to rebuild them? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    logger.info("Skipping index building...")
                    return True
            
            # Import and run the fixed FAISS indexer
            logger.info("Importing fixed FAISS indexer...")
            
            # We need to use the fixed version
            try:
                # Try to import the existing one first
                from complete_faiss_indexer import Complete5GNIDDFAISSIndexer
                logger.info("Using existing complete_faiss_indexer.py")
            except ImportError:
                logger.error("complete_faiss_indexer.py not found!")
                return False
            
            # Create indexer and build indices
            logger.info("Building FAISS indices (this may take a few minutes)...")
            indexer = Complete5GNIDDFAISSIndexer()
            
            # Load data
            datasets = indexer.load_and_preprocess_data()
            if not datasets:
                logger.error("Failed to load datasets!")
                return False
            
            # Build indices
            indexer.build_indices(datasets)
            
            # Save indices
            indexer.save_indices()
            
            logger.info("✓ FAISS indices built successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error building FAISS indices: {e}")
            logger.exception("Full error details:")
            return False
    
    def test_gpu_llm(self):
        """Test the GPU LLM client."""
        logger.info("=== Testing GPU LLM Client ===")
        
        try:
            # Check if GPU is available first
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("⚠ GPU not available, skipping GPU LLM test")
                    return True
            except ImportError:
                logger.warning("⚠ PyTorch not available, skipping GPU LLM test")
                return True
            
            from gpu_llm_client import test_gpu_llm_client
            
            logger.info("Testing GPU LLM client (this may take 30-60 seconds)...")
            success = test_gpu_llm_client()
            
            if success:
                logger.info("✓ GPU LLM client test passed")
                return True
            else:
                logger.warning("⚠ GPU LLM client test failed, but continuing...")
                return True  # Don't fail setup for LLM issues
                
        except Exception as e:
            logger.warning(f"GPU LLM client test failed: {e}")
            logger.warning("Continuing with setup anyway...")
            return True  # Don't fail setup for LLM issues
    
    def run_quick_test(self):
        """Run a quick test of the system."""
        logger.info("=== Running Quick System Test ===")
        
        try:
            from complete_faiss_indexer import Complete5GNIDDFAISSIndexer
            
            # Test loading indices
            indexer = Complete5GNIDDFAISSIndexer()
            indexer.load_indices()
            
            # Test classification
            test_flow = {
                'Seq': 1000, 'Dur': 0.1, 'Proto': 'tcp', 'TotPkts': 10,
                'SrcBytes': 500, 'TotBytes': 1000, 'State': 'CON'
            }
            
            classification, confidence, attack_type = indexer.classify_flow(test_flow)
            logger.info(f"Test classification: {classification} (confidence: {confidence:.3f})")
            
            logger.info("✓ Quick test passed")
            return True
            
        except Exception as e:
            logger.error(f"Quick test failed: {e}")
            return False
    
    def run_cli_demo(self, duration=60):
        """Run a quick CLI demo."""
        logger.info("=== Running CLI Demo ===")
        
        try:
            from real_time_detector import main as run_detector
            
            # Temporarily modify sys.argv for the demo
            original_argv = sys.argv.copy()
            sys.argv = [
                "real_time_detector.py",
                "--mode", "demo",
                "--duration", str(duration)
            ]
            
            logger.info(f"Running detection system for {duration} seconds...")
            result = run_detector()
            
            # Restore original argv
            sys.argv = original_argv
            
            if result == 0:
                logger.info("✓ CLI demo completed successfully")
                return True
            else:
                logger.error("✗ CLI demo failed")
                return False
                
        except Exception as e:
            logger.error(f"Error running CLI demo: {e}")
            return False
    
    def setup_complete_system(self):
        """Run the complete setup process."""
        logger.info("🛡️  Network Intrusion Detection System Setup (Fixed)")
        logger.info("=" * 60)
        
        # Step 1: Check requirements
        if not self.check_requirements():
            logger.error("❌ Requirements check failed")
            return False
        
        # Step 2: Install dependencies
        if not self.install_dependencies():
            logger.error("❌ Dependency installation failed")
            return False
        
        # Step 3: Build FAISS indices
        if not self.build_faiss_indices():
            logger.error("❌ FAISS index building failed")
            return False
        
        # Step 4: Run quick test
        if not self.run_quick_test():
            logger.error("❌ Quick test failed")
            return False
        
        # Step 5: Test GPU LLM (optional)
        self.test_gpu_llm()  # Don't fail if this doesn't work
        
        logger.info("✅ Setup completed successfully!")
        logger.info("")
        logger.info("🎉 You can now run:")
        logger.info("   - Web interface: python demo_web_interface.py")
        logger.info("   - CLI demo: python real_time_detector.py --mode demo --duration 30")
        logger.info("   - Or use: python setup_and_run.py --web")
        
        return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Fixed Network IDS Setup and Demo")
    parser.add_argument("--setup", action="store_true",
                       help="Run complete setup process")
    parser.add_argument("--inspect", action="store_true",
                       help="Inspect data structure only")
    parser.add_argument("--demo", type=int, default=0,
                       help="Run CLI demo for specified seconds")
    parser.add_argument("--web", action="store_true",
                       help="Launch web interface")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick system test")
    
    args = parser.parse_args()
    
    setup = FixedNetworkIDSSetup()
    
    try:
        if args.inspect:
            # Just inspect data structure
            setup.inspect_data_structure()
            
        elif args.setup:
            # Run complete setup
            success = setup.setup_complete_system()
            if not success:
                logger.error("Setup failed!")
                return 1
                
        elif args.demo > 0:
            # Run CLI demo
            if not setup.check_requirements():
                logger.error("Requirements not met!")
                return 1
            success = setup.run_cli_demo(duration=args.demo)
            if not success:
                return 1
                
        elif args.web:
            # Launch web interface
            if not setup.check_requirements():
                logger.error("Requirements not met!")
                return 1
            
            logger.info("Launching web interface...")
            from demo_web_interface import app, socketio
            socketio.run(app, debug=False, host='0.0.0.0', port=5000)
            
        elif args.quick_test:
            # Quick test
            logger.info("Running quick system test...")
            if not setup.check_requirements():
                return 1
            if not setup.run_quick_test():
                return 1
            logger.info("✅ Quick test passed!")
            
        else:
            # Interactive mode
            logger.info("🛡️  Network Intrusion Detection System (Fixed)")
            logger.info("=" * 50)
            logger.info("Choose an option:")
            logger.info("1. Inspect data structure")
            logger.info("2. Complete setup (recommended for first run)")
            logger.info("3. Run CLI demo (30 seconds)")
            logger.info("4. Launch web interface")
            logger.info("5. Quick system test")
            logger.info("6. Exit")
            
            while True:
                try:
                    choice = input("\nEnter your choice (1-6): ").strip()
                    
                    if choice == "1":
                        setup.inspect_data_structure()
                        break
                        
                    elif choice == "2":
                        success = setup.setup_complete_system()
                        if success:
                            logger.info("\n🎉 Setup complete! You can now:")
                            logger.info("   - Run: python demo_web_interface.py")
                            logger.info("   - Or: python setup_and_run.py --web")
                        break
                        
                    elif choice == "3":
                        if setup.check_requirements():
                            setup.run_cli_demo(duration=30)
                        break
                        
                    elif choice == "4":
                        if setup.check_requirements():
                            logger.info("Launching web interface...")
                            from demo_web_interface import app, socketio
                            socketio.run(app, debug=False, host='0.0.0.0', port=5000)
                        break
                        
                    elif choice == "5":
                        if setup.check_requirements() and setup.run_quick_test():
                            logger.info("✅ Quick test passed!")
                        break
                        
                    elif choice == "6":
                        logger.info("Goodbye!")
                        break
                        
                    else:
                        logger.warning("Invalid choice. Please enter 1-6.")
                        
                except KeyboardInterrupt:
                    logger.info("\nGoodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error: {e}")
                    break
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())