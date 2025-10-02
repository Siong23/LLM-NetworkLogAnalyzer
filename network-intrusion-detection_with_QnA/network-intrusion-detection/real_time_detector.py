# real_time_detector.py - CORRECTED VERSION WITH GPU SUPPORT
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
        proto_map = {'TCP': 'tcp', 'UDP': 'udp', 'ICMP': 'icmp'}
        proto = proto_map.get(self.protocol.upper(), 'tcp')

        state_map = {
            'SYN': 'REQ',
            'SYN+ACK': 'CON',
            'ACK': 'CON',
            'FIN': 'FIN',
            'RST': 'RST',
            'PSH': 'CON',
            'ICMP': 'ECO'
        }
        state = state_map.get(self.tcp_flags, 'CON')

        mean_pkt_sz = self.bytes_count / max(self.packets_count, 1)

        features = {
            'Seq': hash(f"{self.src_ip}{self.dst_ip}{self.timestamp}") % 1000000,
            'Dur': self.duration, 'RunTime': self.duration, 'Mean': self.duration,
            'Sum': self.duration, 'Min': self.duration, 'Max': self.duration,
            'Proto': proto,
            'sTos': 0.0, 'dTos': 0.0, 'sDSb': 'cs0', 'dDSb': 'cs0',
            'sTtl': random.randint(50, 255), 'dTtl': random.randint(50, 255),
            'sHops': random.randint(1, 20), 'dHops': random.randint(1, 20),
            'Cause': 'Start', 'State': state,
            'TotPkts': self.packets_count, 'SrcPkts': max(1, self.packets_count // 2),
            'DstPkts': max(1, self.packets_count - (self.packets_count // 2)),
            'TotBytes': self.bytes_count, 'SrcBytes': max(1, self.bytes_count // 2),
            'DstBytes': max(1, self.bytes_count - (self.bytes_count // 2)),
            'Offset': random.randint(1000, 200000),
            'sMeanPktSz': mean_pkt_sz, 'dMeanPktSz': mean_pkt_sz,
            'Load': 0.0, 'SrcLoad': 0.0, 'DstLoad': 0.0,
            'Loss': 0, 'SrcLoss': 0, 'DstLoss': 0, 'pLoss': 0.0,
            'SrcGap': 0.0, 'DstGap': 0.0,
            'Rate': self.bytes_count / max(self.duration, 0.001),
            'SrcRate': (self.bytes_count / 2) / max(self.duration, 0.001),
            'DstRate': (self.bytes_count / 2) / max(self.duration, 0.001),
            'SrcWin': 0, 'DstWin': 0, 'sVid': 0, 'dVid': 0,
            'SrcTCPBase': 0, 'DstTCPBase': 0,
            'TcpRtt': self.duration * 1000 if proto == 'tcp' else 0.0,
            'SynAck': 0.0, 'AckDat': self.bytes_count if proto == 'tcp' else 0.0
        }
        return features

@dataclass
class QuestionRequest:
    """Represents a question about a specific analysis."""
    question_id: str
    question: str
    original_analysis: Dict
    flow_id: str
    timestamp: str
    requester_id: Optional[str] = None

class RealTimeDetectionSystem:
    """Enhanced real-time network flow detection and analysis system with question capability."""

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

        # Initialize LLM client ONCE here
        if self.use_gpu:
            # Check GPU availability first
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("Initializing GPU-optimized LLM client...")
                    self.llm_client = GPUDeepSeekLLMClient()
                else:
                    logger.warning("GPU not available, falling back to basic mode")
                    self.use_gpu = False
                    self.llm_client = None
            except ImportError:
                logger.warning("PyTorch not available, falling back to basic mode")
                self.use_gpu = False
                self.llm_client = None
        else:
            logger.info("GPU disabled by configuration")
            self.llm_client = None

        self.analyzed_attack_types = set()
        self.stats = {"benign": 0, "suspicious": 0, "malicious": 0, "analyzed": 0, "questions_answered": 0}
        self.demo_flow_counter = 0
        self.demo_pattern = self._create_demo_pattern()
        
        # Enhanced queue system
        self.flow_queue = queue.Queue(maxsize=1000)
        self.analysis_queue = queue.Queue(maxsize=100)
        self.question_queue = queue.PriorityQueue(maxsize=50)  # Priority queue for questions
        
        # Analysis state
        self.analysis_paused = False
        self.current_question = None
        
        # Event system
        self.stop_event = threading.Event()
        self.question_complete_event = threading.Event()
        
        self._load_indices()
        self.start_time = None
        self.total_flows_processed = 0

        # Storage for analysis results (for questions)
        self.analysis_storage = {}  # flow_id -> analysis_dict

    def _create_demo_pattern(self) -> List[str]:
        """Create a demo pattern that produces realistic classification results."""
        if not self.demo_mode:
            return []
        
        # Pattern designed to produce varied but realistic classifications
        pattern = [
            'benign', 'benign', 'benign', 'benign',  # Start with normal traffic
            'synscan',  # First attack - should be detected
            'benign', 'benign',
            'icmpflood',  # Second attack type
            'benign', 'benign', 'benign',
            'httpflood',  # Third attack type
            'benign', 'benign',
            'udpflood',  # Fourth attack type
            'benign', 'synflood',
            'benign', 'benign', 'tcpconnectscan',
            'benign', 'udpscan', 'benign',
            'benign', 'slowratedos', 'benign'
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
            raise

    def start(self):
        logger.info(f"Starting detection system in {'demo' if self.demo_mode else 'production'} mode")
        logger.info(f"Using {'GPU' if self.use_gpu else 'CPU'} for LLM inference")
        self.start_time = time.time()
        self.stop_event.clear()
        
        # Start worker threads
        self.classification_thread = threading.Thread(target=self._classification_worker, daemon=True)
        self.analysis_thread = threading.Thread(target=self._enhanced_analysis_worker, daemon=True)
        self.simulation_thread = threading.Thread(target=self._simulate_network_flows, daemon=True)
        
        self.classification_thread.start()
        self.analysis_thread.start()
        self.simulation_thread.start()
        
        logger.info("Detection system started successfully")

    def stop(self):
        logger.info("Stopping detection system...")
        self.stop_event.set()
        
        # Signal any waiting questions to complete
        self.question_complete_event.set()
        
        for thread in [self.classification_thread, self.analysis_thread, self.simulation_thread]:
            if hasattr(self, thread.name) and thread.is_alive():
                thread.join(timeout=2)

        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"System ran for {total_time:.2f} seconds")
            logger.info(f"Processed {self.total_flows_processed} flows")

        if self.use_gpu and hasattr(self, 'llm_client') and self.llm_client:
            try:
                self.llm_client.cleanup()
            except Exception as e:
                logger.warning(f"GPU cleanup warning: {e}")

    def ask_question(self, question: str, flow_id: str, requester_id: str = None) -> str:
        """
        Ask a question about a specific analysis.
        
        Args:
            question: The question to ask
            flow_id: The ID of the flow/analysis to ask about
            requester_id: Optional ID of the requester
            
        Returns:
            question_id: Unique identifier for tracking the question
        """
        if flow_id not in self.analysis_storage:
            raise ValueError(f"No analysis found for flow_id: {flow_id}")
        
        question_id = f"q_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        question_request = QuestionRequest(
            question_id=question_id,
            question=question,
            original_analysis=self.analysis_storage[flow_id],
            flow_id=flow_id,
            timestamp=datetime.now().isoformat(),
            requester_id=requester_id
        )
        
        # Add to priority queue (priority 0 = highest priority)
        try:
            self.question_queue.put((0, question_request), timeout=1)
            logger.info(f"❓ Question queued: {question_id} - {question[:50]}...")
            return question_id
        except queue.Full:
            raise RuntimeError("Question queue is full. Please try again later.")

    def pause_analysis(self):
        """Pause normal analysis processing to prioritize questions."""
        self.analysis_paused = True
        logger.info("⏸️ Analysis paused for question processing")

    def resume_analysis(self):
        """Resume normal analysis processing."""
        self.analysis_paused = False
        self.question_complete_event.set()
        logger.info("▶️ Analysis resumed")

    def _simulate_network_flows(self):
        flow_count = 0
        while not self.stop_event.is_set():
            try:
                flow = self._generate_demo_flow() if self.demo_mode else self._generate_realistic_flow()
                flow_count += 1
                if not self.flow_queue.full():
                    self.flow_queue.put(flow)
                else:
                    logger.warning("Flow queue full, dropping flow")
                time.sleep(random.uniform(1.5, 3.0) if self.demo_mode else random.uniform(0.3, 0.8))
                if flow_count % (20 if self.demo_mode else 50) == 0:
                    logger.info(f"Generated {flow_count} flows, queue size: {self.flow_queue.qsize()}")
            except Exception as e:
                logger.error(f"Error in flow simulation: {e}")
                time.sleep(1)

    def _generate_demo_flow(self) -> NetworkFlow:
        """Generate a demo flow that should produce realistic classifications."""
        pattern_index = self.demo_flow_counter % len(self.demo_pattern)
        desired_type = self.demo_pattern[pattern_index]
        self.demo_flow_counter += 1
        logger.info(f"Demo flow #{self.demo_flow_counter}: generating {desired_type} flow")

        # Create base flow with realistic characteristics
        flow = NetworkFlow(
            timestamp=datetime.now().isoformat(),
            src_ip=f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
            dst_ip=f"10.0.{random.randint(1, 254)}.{random.randint(1, 254)}",
            src_port=random.randint(1024, 65535),
            dst_port=random.choice([80, 443, 22, 21, 25, 53, 3389, 8080, 135, 139]),
            protocol=random.choice(['TCP', 'UDP', 'ICMP']),
            bytes_count=random.randint(64, 1500),
            packets_count=random.randint(1, 10),
            duration=random.uniform(0.001, 1.0),
            tcp_flags=random.choice(['SYN', 'ACK', 'PSH', 'FIN', 'RST', 'SYN+ACK'])
        )

        # Modify flow characteristics ONLY if it's an attack type
        if desired_type != 'benign' and desired_type in self.faiss_indexer.attack_types:
            flow.attack_type = desired_type
            self._modify_flow_for_attack_type(flow, desired_type)

        return flow

    def _generate_realistic_flow(self) -> NetworkFlow:
        flow_type = random.choices(['benign', 'malicious', 'suspicious'], weights=[0.8, 0.15, 0.05])[0]
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
            attack_type = random.choice(self.faiss_indexer.attack_types)
            base_flow.attack_type = attack_type
            self._modify_flow_for_attack_type(base_flow, attack_type)
        return base_flow

    def _modify_flow_for_attack_type(self, flow: NetworkFlow, attack_type: str):
        """Modify flow characteristics to match attack patterns that FAISS will recognize."""
        if 'icmpflood' in attack_type.lower():
            flow.protocol = 'ICMP'
            flow.packets_count = random.randint(100, 1000)  # High packet count
            flow.bytes_count = flow.packets_count * 42  # ICMP packet size
            flow.duration = random.uniform(0.001, 0.1)  # Very short duration
            flow.tcp_flags = 'ICMP'
        elif 'udpflood' in attack_type.lower():
            flow.protocol = 'UDP'
            flow.packets_count = random.randint(500, 2000)  # Very high packet count
            flow.bytes_count = random.randint(10000, 100000)  # Large byte count
            flow.duration = random.uniform(0.1, 2.0)
        elif 'synflood' in attack_type.lower():
            flow.protocol = 'TCP'
            flow.tcp_flags = 'SYN'
            flow.packets_count = random.randint(100, 1000)
            flow.bytes_count = flow.packets_count * 64  # SYN packet size
            flow.duration = random.uniform(0.01, 0.5)
        elif 'httpflood' in attack_type.lower():
            flow.protocol = 'TCP'
            flow.dst_port = random.choice([80, 443, 8080])
            flow.packets_count = random.randint(50, 200)
            flow.bytes_count = random.randint(5000, 50000)
            flow.duration = random.uniform(0.1, 2.0)
        elif 'scan' in attack_type.lower():
            flow.protocol = 'TCP'
            flow.tcp_flags = 'SYN'
            flow.dst_port = random.randint(1, 1024)
            flow.packets_count = 1
            flow.bytes_count = 64
            flow.duration = random.uniform(0.001, 0.1)

    def _classification_worker(self):
        processed_count = 0
        while not self.stop_event.is_set():
            try:
                flow = self.flow_queue.get(timeout=1.0)
                features = flow.to_feature_dict()
                classification, confidence, attack_type = self.faiss_indexer.classify_flow(features)
                flow.classification = classification
                flow.confidence = confidence
                if attack_type:
                    flow.attack_type = attack_type

                self.stats[classification] += 1
                processed_count += 1
                self.total_flows_processed += 1

                if classification == 'malicious':
                    self._handle_malicious_flow(flow)

                if self.demo_mode or processed_count % 10 == 0:
                    logger.info(f"Flow {processed_count}: {classification} (conf: {confidence:.3f}, type: {attack_type})")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in classification worker: {e}")
                time.sleep(0.1)

    def _handle_malicious_flow(self, flow: NetworkFlow):
        if self.demo_mode:
            if flow.attack_type in self.analyzed_attack_types:
                return
            self.analyzed_attack_types.add(flow.attack_type)
            logger.info(f"🚨 Queuing {flow.attack_type} for LLM analysis (first of this type in demo)")
        if not self.analysis_queue.full():
            self.analysis_queue.put(flow)
        else:
            logger.warning("Analysis queue full, dropping malicious flow for LLM analysis")

    def _enhanced_analysis_worker(self):
        """Enhanced analysis worker that handles both regular analysis and questions."""
        while not self.stop_event.is_set():
            try:
                # Check for priority questions first
                if not self.question_queue.empty():
                    try:
                        priority, question_request = self.question_queue.get_nowait()
                        self._process_question(question_request)
                        continue
                    except queue.Empty:
                        pass
                
                # If analysis is paused, wait for resume signal
                if self.analysis_paused:
                    logger.info("⏸️ Analysis worker paused, waiting for resume...")
                    self.question_complete_event.wait(timeout=1.0)
                    self.question_complete_event.clear()
                    continue
                
                # Process regular analysis (only if LLM client is available)
                if self.llm_client:
                    try:
                        flow = self.analysis_queue.get(timeout=1.0)
                        self._process_flow_analysis(flow)
                    except queue.Empty:
                        continue
                else:
                    # If no LLM client, just simulate analysis
                    try:
                        flow = self.analysis_queue.get(timeout=1.0)
                        self._simulate_analysis(flow)
                    except queue.Empty:
                        continue
                    
            except Exception as e:
                logger.error(f"Error in enhanced analysis worker: {e}")
                time.sleep(0.2)

    def _process_question(self, question_request: QuestionRequest):
        """Process a question about a specific analysis."""
        try:
            logger.info(f"❓ Processing question: {question_request.question_id}")
            
            # Pause normal analysis while processing question
            was_paused = self.analysis_paused
            self.analysis_paused = True
            
            # Use the LLM client to answer the question (if available)
            if self.llm_client:
                result = self.llm_client.answer_question(
                    question_request.original_analysis,
                    question_request.question,
                    question_request.flow_id
                )
            else:
                # Fallback answer if no LLM client
                result = {
                    "question": question_request.question,
                    "answer": "LLM not available for question answering. The analysis shows this is a network security threat that requires immediate attention.",
                    "response_source": "fallback_answer",
                    "inference_time": 0.1
                }
            
            # Add question metadata
            result.update({
                "question_id": question_request.question_id,
                "flow_id": question_request.flow_id,
                "requester_id": question_request.requester_id
            })
            
            # Emit the result (will be implemented in web interface)
            self._emit_question_result(result)
            
            self.stats["questions_answered"] += 1
            logger.info(f"✅ Question answered: {question_request.question_id}")
            
            # Resume analysis if it wasn't already paused
            if not was_paused:
                self.analysis_paused = False
                
        except Exception as e:
            logger.error(f"❌ Error processing question {question_request.question_id}: {e}")
            error_result = {
                "question_id": question_request.question_id,
                "flow_id": question_request.flow_id,
                "answer": f"Error processing question: {str(e)}",
                "error": True
            }
            self._emit_question_result(error_result)

    def _process_flow_analysis(self, flow: NetworkFlow):
        """Process regular flow analysis."""
        attack_type = flow.attack_type or "unknown"
        features = flow.to_feature_dict()

        logger.info(f"Starting GPU LLM analysis for {attack_type}")
        
        # Use the already initialized LLM client
        result = self.llm_client.analyze(attack_type, features)
        
        # Generate flow ID and store analysis
        flow_id = result.get('flow_id', f"{attack_type}_{int(time.time())}")
        self.analysis_storage[flow_id] = result
        
        # Clean up old analyses (keep only last 100)
        if len(self.analysis_storage) > 100:
            oldest_keys = sorted(self.analysis_storage.keys())[:50]
            for key in oldest_keys:
                del self.analysis_storage[key]
        
        self.stats["analyzed"] += 1
        logger.info(f"LLM analysis completed for {attack_type}")
        
        # Emit the result (will be implemented in web interface)
        self._emit_analysis_result(flow, result)

    def _simulate_analysis(self, flow: NetworkFlow):
        """Simulate analysis when LLM client is not available."""
        attack_type = flow.attack_type or "unknown"
        
        # Create simulated analysis result
        result = {
            "threat_level": "HIGH",
            "risk_score": random.randint(70, 95),
            "attack_vector": f"{attack_type} attack detected",
            "potential_impact": f"Potential {attack_type} network security threat",
            "immediate_actions": [
                f"Block source IP {flow.src_ip}",
                "Monitor for similar patterns",
                "Alert security team"
            ],
            "technical_analysis": f"Simulated analysis for {attack_type} attack pattern",
            "flow_id": f"{attack_type}_{int(time.time())}",
            "rag_assisted": False,
            "response_source": "simulated_analysis"
        }
        
        # Store analysis
        self.analysis_storage[result["flow_id"]] = result
        
        # Clean up old analyses
        if len(self.analysis_storage) > 100:
            oldest_keys = sorted(self.analysis_storage.keys())[:50]
            for key in oldest_keys:
                del self.analysis_storage[key]
        
        self.stats["analyzed"] += 1
        logger.info(f"Simulated analysis completed for {attack_type}")
        
        # Emit the result
        self._emit_analysis_result(flow, result)

    def _emit_question_result(self, result: Dict):
        """Emit question result. Override in web interface."""
        pass

    def _emit_analysis_result(self, flow: NetworkFlow, result: Dict):
        """Emit analysis result. Override in web interface."""
        pass

    def get_stats(self) -> Dict:
        stats = self.stats.copy()
        if self.start_time:
            stats['uptime_seconds'] = time.time() - self.start_time
            total_flows = sum(self.stats.values()) - self.stats.get("analyzed", 0) - self.stats.get("questions_answered", 0)
            if total_flows > 0 and stats['uptime_seconds'] > 0:
                stats['flows_per_second'] = total_flows / stats['uptime_seconds']
        
        # Add queue status
        stats.update({
            'flow_queue_size': self.flow_queue.qsize(),
            'analysis_queue_size': self.analysis_queue.qsize(),
            'question_queue_size': self.question_queue.qsize(),
            'analysis_paused': self.analysis_paused,
            'stored_analyses': len(self.analysis_storage),
            'gpu_enabled': self.use_gpu,
            'llm_available': self.llm_client is not None
        })
        
        return stats

    def get_available_analyses(self) -> List[Dict]:
        """Get list of available analyses for questioning."""
        analyses = []
        for flow_id, analysis in self.analysis_storage.items():
            analyses.append({
                'flow_id': flow_id,
                'attack_type': analysis.get('attack_vector', 'Unknown'),
                'threat_level': analysis.get('threat_level', 'Unknown'),
                'timestamp': analysis.get('analysis_timestamp', 'Unknown'),
                'rag_assisted': analysis.get('rag_assisted', False)
            })
        return sorted(analyses, key=lambda x: x['timestamp'], reverse=True)


def main():
    """Main function for standalone demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Network Intrusion Detection System")
    parser.add_argument("--mode", choices=["demo", "production"], default="demo",
                       help="Detection mode")
    parser.add_argument("--duration", type=int, default=60,
                       help="Duration to run in seconds")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    try:
        detector = RealTimeDetectionSystem(
            demo_mode=(args.mode == "demo"),
            use_gpu=not args.no_gpu
        )
        
        detector.start()
        
        logger.info(f"Running detection system for {args.duration} seconds...")
        time.sleep(args.duration)
        
        detector.stop()
        
        # Print final stats
        stats = detector.get_stats()
        logger.info("Final Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())