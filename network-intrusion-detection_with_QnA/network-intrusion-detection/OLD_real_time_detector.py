import asyncio
import json
import time
import random
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import threading
import queue
from complete_faiss_indexer import Complete5GNIDDFAISSIndexer
from gpu_llm_client import GPUDeepSeekLLMClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NetworkFlow:
    """Represents a network flow with all its features matching the 5G-NIDD dataset structure."""
    timestamp: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    bytes_count: int
    packets_count: int
    duration: float
    tcp_flags: str
    classification: Optional[str] = None
    confidence: Optional[float] = None
    attack_type: Optional[str] = None
    llm_analysis: Optional[Dict] = None
    
    def to_feature_dict(self) -> Dict:
        """Convert flow to feature dictionary matching the 5G-NIDD dataset structure."""
        
        # Map protocol to match dataset
        proto_map = {'TCP': 'tcp', 'UDP': 'udp', 'ICMP': 'icmp'}
        proto = proto_map.get(self.protocol.upper(), 'tcp')
        
        # Map TCP flags to state
        state_map = {
            'SYN': 'REQ',
            'SYN+ACK': 'CON', 
            'ACK': 'CON',
            'FIN': 'FIN',
            'RST': 'RST',
            'PSH': 'CON'
        }
        state = state_map.get(self.tcp_flags, 'CON')
        
        # Calculate mean packet size
        mean_pkt_sz = self.bytes_count / max(self.packets_count, 1)
        
        # Generate realistic feature values based on the actual dataset structure
        features = {
            # Core identification features
            'Seq': hash(f"{self.src_ip}{self.dst_ip}{self.timestamp}") % 1000000,
            'Dur': self.duration,
            'RunTime': self.duration,
            'Mean': self.duration,
            'Sum': self.duration,
            'Min': self.duration,
            'Max': self.duration,
            
            # Protocol and networking features
            'Proto': proto,
            'sTos': 0.0,
            'dTos': 0.0,
            'sDSb': 'cs0',  # Default DSCP marking
            'dDSb': 'cs0',
            'sTtl': random.randint(50, 255),  # Typical TTL values
            'dTtl': random.randint(50, 255),
            'sHops': random.randint(1, 20),
            'dHops': random.randint(1, 20),
            
            # Flow state
            'Cause': 'Start',
            'State': state,
            
            # Packet and byte statistics
            'TotPkts': self.packets_count,
            'SrcPkts': max(1, self.packets_count // 2),
            'DstPkts': max(1, self.packets_count - (self.packets_count // 2)),
            'TotBytes': self.bytes_count,
            'SrcBytes': max(1, self.bytes_count // 2),
            'DstBytes': max(1, self.bytes_count - (self.bytes_count // 2)),
            
            # Offset and packet size features
            'Offset': random.randint(1000, 200000),
            'sMeanPktSz': mean_pkt_sz,
            'dMeanPktSz': mean_pkt_sz,
            
            # Load and loss features
            'Load': 0.0,
            'SrcLoad': 0.0,
            'DstLoad': 0.0,
            'Loss': 0,
            'SrcLoss': 0,
            'DstLoss': 0,
            'pLoss': 0.0,
            
            # Gap and rate features
            'SrcGap': 0.0,
            'DstGap': 0.0,
            'Rate': self.bytes_count / max(self.duration, 0.001),
            'SrcRate': (self.bytes_count / 2) / max(self.duration, 0.001),
            'DstRate': (self.bytes_count / 2) / max(self.duration, 0.001),
            
            # Window and TCP features
            'SrcWin': 0,
            'DstWin': 0,
            'sVid': 0,
            'dVid': 0,
            'SrcTCPBase': 0,
            'DstTCPBase': 0,
            'TcpRtt': self.duration * 1000 if proto == 'tcp' else 0.0,
            'SynAck': 0.0,
            'AckDat': self.bytes_count if proto == 'tcp' else 0.0
        }
        
        return features

class RealTimeDetectionSystem:
    """Real-time network flow detection and analysis system with GPU LLM."""
    
    def __init__(self, demo_mode: bool = True, use_gpu: bool = True):
        """
        Initialize the detection system.
        
        Args:
            demo_mode: If True, limits LLM analysis to one per attack type
            use_gpu: If True, uses GPU-optimized LLM client
        """
        self.demo_mode = demo_mode
        self.use_gpu = use_gpu
        self.faiss_indexer = Complete5GNIDDFAISSIndexer()
        
        # Initialize LLM client
        if self.use_gpu:
            logger.info("Initializing GPU-optimized LLM client...")
            try:
                self.llm_client = GPUDeepSeekLLMClient()
            except Exception as e:
                logger.error(f"Failed to initialize GPU LLM client: {e}")
                logger.info("Falling back to CPU mode...")
                self.use_gpu = False
                raise NotImplementedError("CPU fallback not implemented in this version")
        else:
            logger.info("GPU disabled, CPU mode not implemented")
            raise NotImplementedError("CPU fallback not implemented in this version")
        
        # Tracking for demo mode
        self.analyzed_attack_types = set()
        
        # Statistics
        self.stats = {
            "benign": 0,
            "suspicious": 0,
            "malicious": 0,
            "analyzed": 0
        }
        
        # Demo flow scheduling
        self.demo_flow_counter = 0
        self.demo_pattern = self._create_demo_pattern()
        
        # Queues for processing
        self.flow_queue = queue.Queue(maxsize=1000)
        self.analysis_queue = queue.Queue(maxsize=100)
        
        # Event for stopping the system
        self.stop_event = threading.Event()
        
        # Load FAISS indices
        self._load_indices()
        
        # Performance tracking
        self.start_time = None
        self.total_flows_processed = 0
    
    def _create_demo_pattern(self) -> List[str]:
        """Create an engaging demo pattern that shows variety quickly."""
        if not self.demo_mode:
            return []
        
        # Demo pattern: Start with a few benign, then mix in attacks
        pattern = [
            'benign', 'benign', 'benign',  # First 3 benign to establish baseline
            'synscan',                      # Port scan attack
            'benign', 'benign',            # A couple more benign
            'icmpflood',                   # ICMP flood attack  
            'benign',                      # One benign
            'httpflood',                   # HTTP flood attack
            'benign', 'benign',            # Two benign
            'udpflood',                    # UDP flood attack
            'benign',                      # One benign
            'synflood',                    # SYN flood attack
            'benign', 'suspicious',        # Benign then suspicious
            'tcpconnectscan',              # TCP connect scan
            'benign',                      # One benign
            'udpscan',                     # UDP scan
            'suspicious',                  # Another suspicious
            'benign', 'benign',            # Two benign
            # Pattern repeats but with different attack types mixed in
            'httpflood', 'benign', 'synscan', 'benign', 'icmpflood', 
            'suspicious', 'benign', 'udpflood', 'benign', 'synflood'
        ]
        
        logger.info(f"Demo pattern created with {len(pattern)} flow types")
        return pattern
    
    def _load_indices(self):
        """Load the FAISS indices."""
        try:
            self.faiss_indexer.load_indices()
            logger.info("FAISS indices loaded successfully")
            logger.info(f"Available attack types: {self.faiss_indexer.attack_types}")
        except Exception as e:
            logger.error(f"Failed to load FAISS indices: {e}")
            logger.info("Please run the complete FAISS indexer first to build indices")
            raise
    
    def start(self):
        """Start the real-time detection system."""
        logger.info(f"Starting detection system in {'demo' if self.demo_mode else 'production'} mode")
        logger.info(f"Using {'GPU' if self.use_gpu else 'CPU'} for LLM inference")
        
        self.start_time = time.time()
        self.stop_event.clear()
        
        # Start worker threads
        self.classification_thread = threading.Thread(target=self._classification_worker, daemon=True)
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.simulation_thread = threading.Thread(target=self._simulate_network_flows, daemon=True)
        
        self.classification_thread.start()
        self.analysis_thread.start()
        self.simulation_thread.start()
        
        logger.info("Detection system started successfully")
    
    def stop(self):
        """Stop the detection system."""
        logger.info("Stopping detection system...")
        self.stop_event.set()
        
        # Wait for threads to finish (with timeout)
        for thread in [self.classification_thread, self.analysis_thread, self.simulation_thread]:
            if hasattr(self, thread.name) and thread.is_alive():
                thread.join(timeout=2)
        
        # Print final performance stats
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"System ran for {total_time:.2f} seconds")
            logger.info(f"Processed {self.total_flows_processed} flows")
            
            if self.use_gpu and hasattr(self, 'llm_client'):
                try:
                    gpu_stats = self.llm_client.get_performance_stats()
                    logger.info(f"GPU LLM Performance: {gpu_stats}")
                except Exception as e:
                    logger.warning(f"Could not get GPU stats: {e}")
        
        # Cleanup GPU resources
        if self.use_gpu and hasattr(self, 'llm_client'):
            try:
                self.llm_client.cleanup()
            except Exception as e:
                logger.warning(f"GPU cleanup warning: {e}")
    
    def _simulate_network_flows(self):
        """Simulate network flows with demo-friendly patterns."""
        flow_count = 0
        
        while not self.stop_event.is_set():
            try:
                # Generate flow based on mode
                if self.demo_mode:
                    flow = self._generate_demo_flow()
                else:
                    flow = self._generate_realistic_flow()
                
                flow_count += 1
                
                # Add to processing queue
                if not self.flow_queue.full():
                    self.flow_queue.put(flow)
                else:
                    logger.warning("Flow queue full, dropping flow")
                
                # Demo timing - faster for more excitement
                if self.demo_mode:
                    time.sleep(random.uniform(1.5, 3.0))  # Demo: 1.5-3s intervals (faster)
                else:
                    time.sleep(random.uniform(0.3, 0.8))  # Production: 0.3-0.8s intervals
                
                # Log progress every 20 flows in demo mode
                if flow_count % (20 if self.demo_mode else 50) == 0:
                    logger.info(f"Generated {flow_count} flows, queue size: {self.flow_queue.qsize()}")
                
            except Exception as e:
                logger.error(f"Error in flow simulation: {e}")
                time.sleep(1)
    
    def _generate_demo_flow(self) -> NetworkFlow:
        """Generate flows following the demo pattern for better demonstration."""
        
        # Use the demo pattern to determine flow type
        pattern_index = self.demo_flow_counter % len(self.demo_pattern)
        desired_type = self.demo_pattern[pattern_index]
        self.demo_flow_counter += 1
        
        logger.info(f"Demo flow #{self.demo_flow_counter}: generating {desired_type} flow")
        
        # Generate base flow
        base_flow = NetworkFlow(
            timestamp=datetime.now().isoformat(),
            src_ip=f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
            dst_ip=f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
            src_port=random.randint(1024, 65535),
            dst_port=random.choice([80, 443, 22, 21, 25, 53, 3389, 8080, 135, 139]),
            protocol=random.choice(['TCP', 'UDP', 'ICMP']),
            bytes_count=random.randint(64, 10000),
            packets_count=random.randint(1, 100),
            duration=random.uniform(0.001, 5.0),
            tcp_flags=random.choice(['SYN', 'ACK', 'PSH', 'FIN', 'RST', 'SYN+ACK'])
        )
        
        if desired_type == 'benign':
            # Keep as normal benign flow
            pass
        elif desired_type == 'suspicious':
            # Make it slightly anomalous but not clearly malicious
            base_flow.bytes_count = random.randint(5000, 15000)
            base_flow.packets_count = random.randint(50, 150)
            base_flow.duration = random.uniform(0.5, 2.0)
        elif desired_type in self.faiss_indexer.attack_types:
            # Generate specific attack type
            base_flow.attack_type = desired_type
            self._modify_flow_for_attack_type(base_flow, desired_type)
        
        return base_flow
    
    def _generate_realistic_flow(self) -> NetworkFlow:
        """Generate flows with realistic distribution for production mode."""
        
        # Production: Realistic distribution
        flow_type = random.choices(['benign', 'malicious', 'suspicious'], 
                                 weights=[0.8, 0.15, 0.05])[0]
        
        # Generate base flow
        base_flow = NetworkFlow(
            timestamp=datetime.now().isoformat(),
            src_ip=f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
            dst_ip=f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
            src_port=random.randint(1024, 65535),
            dst_port=random.choice([80, 443, 22, 21, 25, 53, 3389, 8080, 135, 139]),
            protocol=random.choice(['TCP', 'UDP', 'ICMP']),
            bytes_count=random.randint(64, 10000),
            packets_count=random.randint(1, 100),
            duration=random.uniform(0.001, 5.0),
            tcp_flags=random.choice(['SYN', 'ACK', 'PSH', 'FIN', 'RST', 'SYN+ACK'])
        )
        
        if flow_type == 'malicious':
            # Select from actual attack types in the dataset
            attack_type = random.choice(self.faiss_indexer.attack_types)
            base_flow.attack_type = attack_type
            self._modify_flow_for_attack_type(base_flow, attack_type)
        
        return base_flow
    
    def _modify_flow_for_attack_type(self, flow: NetworkFlow, attack_type: str):
        """Modify flow characteristics based on attack type patterns from the dataset."""
        
        if 'icmpflood' in attack_type.lower():
            flow.protocol = 'ICMP'
            flow.packets_count = random.randint(1, 5)
            flow.bytes_count = 42  # Typical ICMP packet size
            flow.duration = random.uniform(0.001, 0.01)
            flow.tcp_flags = 'ICMP'
            
        elif 'udpflood' in attack_type.lower():
            flow.protocol = 'UDP'
            flow.packets_count = random.randint(100, 1000)
            flow.bytes_count = random.randint(1000, 50000)
            flow.duration = random.uniform(0.1, 2.0)
            
        elif 'synflood' in attack_type.lower():
            flow.protocol = 'TCP'
            flow.tcp_flags = 'SYN'
            flow.packets_count = random.randint(50, 500)
            flow.bytes_count = random.randint(500, 5000)
            flow.duration = random.uniform(0.01, 0.5)
            
        elif 'httpflood' in attack_type.lower():
            flow.protocol = 'TCP'
            flow.dst_port = random.choice([80, 443, 8080])
            flow.packets_count = random.randint(10, 100)
            flow.bytes_count = random.randint(1000, 20000)
            flow.duration = random.uniform(0.1, 2.0)
            
        elif 'scan' in attack_type.lower():
            flow.protocol = 'TCP'
            flow.tcp_flags = 'SYN'
            flow.dst_port = random.randint(1, 1024)
            flow.packets_count = 1
            flow.bytes_count = 64
            flow.duration = random.uniform(0.001, 0.1)
    
    def _classification_worker(self):
        """Worker thread for classifying network flows using FAISS."""
        processed_count = 0
        
        while not self.stop_event.is_set():
            try:
                # Get flow from queue with timeout
                flow = self.flow_queue.get(timeout=1.0)
                
                # Classify the flow using FAISS
                features = flow.to_feature_dict()
                classification, confidence, attack_type = self.faiss_indexer.classify_flow(features)
                
                # Update flow with classification results
                flow.classification = classification
                flow.confidence = confidence
                if attack_type:
                    flow.attack_type = attack_type
                
                # Update statistics
                self.stats[classification] += 1
                processed_count += 1
                self.total_flows_processed += 1
                
                # Send malicious flows for LLM analysis
                if classification == 'malicious':
                    self._handle_malicious_flow(flow)
                
                # Log classification (more verbose in demo mode)
                if self.demo_mode or processed_count % 10 == 0:
                    logger.info(f"Flow {processed_count}: {classification} "
                              f"(conf: {confidence:.3f}, type: {attack_type})")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in classification worker: {e}")
                time.sleep(0.1)
    
    def _handle_malicious_flow(self, flow: NetworkFlow):
        """Handle a detected malicious flow."""
        
        # In demo mode, only analyze one flow per attack type
        if self.demo_mode:
            if flow.attack_type in self.analyzed_attack_types:
                logger.debug(f"Skipping LLM analysis for {flow.attack_type} (demo mode - already analyzed)")
                return
            else:
                self.analyzed_attack_types.add(flow.attack_type)
                logger.info(f"🚨 Queuing {flow.attack_type} for LLM analysis (first of this type in demo)")
        
        # Add to analysis queue
        if not self.analysis_queue.full():
            self.analysis_queue.put(flow)
        else:
            logger.warning("Analysis queue full, dropping malicious flow for LLM analysis")
    
    def _analysis_worker(self):
        """Worker thread for GPU LLM analysis of malicious flows."""
        analysis_count = 0
        
        while not self.stop_event.is_set():
            try:
                # Get malicious flow from queue with timeout
                flow = self.analysis_queue.get(timeout=1.0)
                analysis_count += 1
                
                logger.info(f"🔍 Starting GPU LLM analysis #{analysis_count} for {flow.attack_type} attack "
                           f"from {flow.src_ip} → {flow.dst_ip}")
                
                start_time = time.time()
                
                # Get similar flows for context
                similar_flows = self.faiss_indexer.get_similar_flows(
                    flow.to_feature_dict(), flow.attack_type, k=3
                )
                
                # Analyze with GPU LLM
                analysis = self.llm_client.analyze_malicious_flow(flow, similar_flows)
                
                # Store analysis results
                flow.llm_analysis = analysis
                
                # Update statistics
                self.stats["analyzed"] += 1
                
                analysis_time = time.time() - start_time
                
                logger.info(f"✅ GPU LLM analysis #{analysis_count} completed in {analysis_time:.2f}s")
                logger.info(f"  🎯 Attack: {flow.attack_type}")
                logger.info(f"  ⚠️  Threat Level: {analysis.get('threat_level', 'UNKNOWN')}")
                logger.info(f"  📊 Risk Score: {analysis.get('risk_score', 'N/A')}/100")
                
                # Log GPU memory usage periodically
                if analysis_count % 5 == 0:
                    try:
                        gpu_stats = self.llm_client.get_performance_stats()
                        logger.info(f"🖥️  GPU Stats - Memory: {gpu_stats.get('gpu_memory_allocated_gb', 0):.2f}GB, "
                                   f"Avg Time: {gpu_stats.get('average_inference_time', 0):.2f}s")
                    except Exception as e:
                        logger.warning(f"Could not get GPU stats: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in GPU LLM analysis worker: {e}")
                time.sleep(0.1)
                # Continue processing other flows even if one fails
    
    def get_stats(self) -> Dict:
        """Get current system statistics."""
        stats = self.stats.copy()
        
        # Add performance metrics
        if self.start_time:
            stats['uptime_seconds'] = time.time() - self.start_time
            total_flows = sum(self.stats.values())
            if total_flows > 0 and stats['uptime_seconds'] > 0:
                stats['flows_per_second'] = total_flows / stats['uptime_seconds']
        
        # Add GPU stats if available
        if self.use_gpu and hasattr(self, 'llm_client'):
            try:
                gpu_stats = self.llm_client.get_performance_stats()
                stats['gpu_performance'] = gpu_stats
            except Exception:
                pass  # GPU stats optional
        
        return stats
    
    def get_recent_flows(self, limit: int = 20) -> List[Dict]:
        """Get recent flows (placeholder - implement with actual storage)."""
        # In a real implementation, you would store flows and retrieve them here
        return []

def main():
    """Main function to run the detection system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Network Flow Detection System with GPU LLM")
    parser.add_argument("--mode", choices=["demo", "production"], default="demo",
                       help="Operation mode (demo or production)")
    parser.add_argument("--duration", type=int, default=180,
                       help="Duration to run in seconds (default: 180)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration (use CPU)")
    parser.add_argument("--gpu-memory", type=float, default=7.5,
                       help="Maximum GPU memory to use in GB (default: 7.5)")
    
    args = parser.parse_args()
    
    # Check GPU availability
    use_gpu = not args.no_gpu
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                use_gpu = False
            else:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        except ImportError:
            logger.warning("PyTorch not available, falling back to CPU")
            use_gpu = False
    
    # Initialize and start the detection system
    try:
        detector = RealTimeDetectionSystem(
            demo_mode=(args.mode == "demo"),
            use_gpu=use_gpu
        )
        detector.start()
        
        # Run for specified duration
        logger.info(f"🚀 Running detection system for {args.duration} seconds...")
        logger.info("Press Ctrl+C to stop early")
        
        if args.mode == "demo":
            logger.info("📺 DEMO MODE: Flows will follow an exciting pattern!")
            logger.info("   - First few flows will be benign")
            logger.info("   - Then attacks will start appearing regularly")
            logger.info("   - Each attack type will get LLM analysis (once per type)")
        
        start_time = time.time()
        try:
            while time.time() - start_time < args.duration:
                time.sleep(5)
                
                # Print stats every 30 seconds
                if int(time.time() - start_time) % 30 == 0:
                    stats = detector.get_stats()
                    logger.info(f"📊 Stats: Benign={stats['benign']}, "
                               f"Suspicious={stats['suspicious']}, "
                               f"Malicious={stats['malicious']}, "
                               f"Analyzed={stats['analyzed']}")
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        
    except Exception as e:
        logger.error(f"Failed to initialize detection system: {e}")
        return 1
        
    finally:
        # Stop the system
        if 'detector' in locals():
            detector.stop()
        
        # Print final statistics
        if 'detector' in locals():
            stats = detector.get_stats()
            logger.info("🏁 Final Statistics:")
            for category, count in stats.items():
                if isinstance(count, (int, float)) and not category.endswith('_seconds'):
                    logger.info(f"  {category.capitalize()}: {count}")
    
    return 0

if __name__ == "__main__":
    exit(main())