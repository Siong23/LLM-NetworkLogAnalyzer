#!/usr/bin/env python3
"""
Feature Demonstration Script for Enhanced Network IDS

This script demonstrates the new question-answering and RAG features
of the Network Intrusion Detection System.
"""

import time
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_rag_system():
    """Demonstrate the RAG system capabilities."""
    print("=" * 60)
    print("🔍 RAG SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    try:
        from analysis_rag_index import AnalysisRAGIndex
        
        # Initialize RAG
        rag = AnalysisRAGIndex()
        
        # Add some demo analysis data
        demo_analyses = [
            {
                "text": """SYN Flood Attack Analysis:
                High volume of TCP SYN packets detected without corresponding ACK responses.
                Attack characteristics: 15,000 SYN packets/second from source 192.168.1.100.
                Target exhaustion of connection table on destination server.
                Immediate actions: Block source IP, implement SYN cookies, enable SYN flood protection.
                Investigation: Check for botnet activity, analyze traffic patterns for coordinated attack.
                Prevention: Rate limiting, firewall rules, DDoS mitigation appliance.""",
                "metadata": {
                    "attack_type": "synflood",
                    "threat_level": "HIGH",
                    "risk_score": 90,
                    "protocol": "TCP",
                    "analysis_source": "demo_data"
                }
            },
            {
                "text": """ICMP Flood Attack Analysis:
                Excessive ICMP echo requests causing network congestion and service degradation.
                Attack volume: 50,000 ICMP packets/second, payload size 1472 bytes.
                Network impact: 85% bandwidth utilization, 400ms latency increase.
                Immediate actions: Rate limit ICMP traffic, block source network, alert NOC.
                Investigation: Trace attack origin, check for amplification vectors.
                Prevention: ICMP rate limiting, ingress filtering, network monitoring.""",
                "metadata": {
                    "attack_type": "icmpflood",
                    "threat_level": "HIGH",
                    "risk_score": 85,
                    "protocol": "ICMP",
                    "analysis_source": "demo_data"
                }
            },
            {
                "text": """HTTP Flood Attack Analysis:
                Application layer DDoS targeting web services with legitimate HTTP requests.
                Attack pattern: 10,000 concurrent connections, targeting /search endpoints.
                Server impact: CPU utilization 95%, response time degraded to 15 seconds.
                Immediate actions: Enable rate limiting, implement CAPTCHA, geographic blocking.
                Investigation: Analyze user-agent patterns, check for botnet signatures.
                Prevention: Web application firewall, content delivery network, load balancing.""",
                "metadata": {
                    "attack_type": "httpflood",
                    "threat_level": "MEDIUM",
                    "risk_score": 75,
                    "protocol": "TCP",
                    "analysis_source": "demo_data"
                }
            }
        ]
        
        # Add analyses to RAG
        for analysis in demo_analyses:
            doc_id = rag.add_document(analysis["text"], analysis["metadata"])
            print(f"✅ Added {analysis['metadata']['attack_type']} analysis (ID: {doc_id})")
        
        print(f"\n📊 RAG Statistics:")
        stats = rag.get_stats()
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Attack types: {', '.join(stats['attack_types'])}")
        
        # Demonstrate search capabilities
        print(f"\n🔍 Search Demonstrations:")
        
        test_queries = [
            "SYN flood TCP attack high volume",
            "ICMP flooding network congestion",
            "HTTP application layer DDoS",
            "rate limiting prevention measures",
            "botnet attack analysis"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = rag.search(query, k=2)
            
            for i, result in enumerate(results):
                attack_type = result['metadata']['attack_type']
                score = result['score']
                print(f"  {i+1}. {attack_type} (Score: {score:.3f})")
        
        return True
        
    except ImportError:
        print("❌ RAG system not available (missing dependencies)")
        return False
    except Exception as e:
        print(f"❌ RAG demo failed: {e}")
        return False

def demo_llm_client():
    """Demonstrate the enhanced LLM client capabilities."""
    print("\n" + "=" * 60)
    print("🤖 LLM CLIENT DEMONSTRATION")
    print("=" * 60)
    
    try:
        from gpu_llm_client import GPUDeepSeekLLMClient
        from types import SimpleNamespace
        
        # Check if GPU is available
        try:
            import torch
            if not torch.cuda.is_available():
                print("⚠️  GPU not available, skipping LLM demo")
                return False
        except ImportError:
            print("⚠️  PyTorch not available, skipping LLM demo")
            return False
        
        print("🚀 Initializing GPU LLM Client...")
        llm_client = GPUDeepSeekLLMClient()
        
        # Create demo flow
        demo_flow = SimpleNamespace(
            timestamp=datetime.now().isoformat(),
            src_ip="192.168.1.100",
            dst_ip="10.0.0.50",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            bytes_count=50000,
            packets_count=1000,
            duration=0.5,
            tcp_flags="SYN",
            attack_type="synflood",
            confidence=0.95
        )
        
        print("📊 Analyzing demo SYN flood attack...")
        analysis = llm_client.analyze_malicious_flow(demo_flow)
        
        print("✅ Analysis completed!")
        print(f"  Threat Level: {analysis.get('threat_level', 'Unknown')}")
        print(f"  Risk Score: {analysis.get('risk_score', 'Unknown')}/100")
        print(f"  RAG Assisted: {analysis.get('rag_assisted', False)}")
        print(f"  Flow ID: {analysis.get('flow_id', 'Unknown')}")
        
        # Demonstrate question-answering
        print("\n❓ Demonstrating Question-Answering...")
        
        demo_questions = [
            "What specific indicators suggest this is a SYN flood attack?",
            "How effective would SYN cookies be against this attack?",
            "What would be the business impact if this attack succeeded?"
        ]
        
        for question in demo_questions:
            print(f"\nQ: {question}")
            try:
                answer_result = llm_client.answer_question(analysis, question)
                answer = answer_result.get('answer', 'No answer generated')
                inference_time = answer_result.get('inference_time', 0)
                
                # Truncate long answers for demo
                if len(answer) > 200:
                    answer = answer[:200] + "..."
                
                print(f"A: {answer}")
                print(f"   (Response time: {inference_time:.2f}s)")
            except Exception as e:
                print(f"A: Error - {e}")
        
        # Show performance stats
        print(f"\n📈 Performance Statistics:")
        perf_stats = llm_client.get_performance_stats()
        print(f"  Total analyses: {perf_stats.get('total_analyses', 0)}")
        print(f"  Total questions: {perf_stats.get('total_questions', 0)}")
        print(f"  Average inference time: {perf_stats.get('average_inference_time', 0):.2f}s")
        print(f"  RAG entries: {perf_stats.get('rag_index_size', 0)}")
        
        # Cleanup
        llm_client.cleanup()
        return True
        
    except ImportError:
        print("❌ LLM client not available (missing dependencies)")
        return False
    except Exception as e:
        print(f"❌ LLM demo failed: {e}")
        return False

def demo_detection_system():
    """Demonstrate the enhanced detection system."""
    print("\n" + "=" * 60)
    print("🛡️  DETECTION SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    try:
        from real_time_detector import RealTimeDetectionSystem
        
        print("🚀 Initializing Detection System...")
        detector = RealTimeDetectionSystem(demo_mode=True, use_gpu=False)  # Use CPU for demo
        
        print("📊 System capabilities:")
        print(f"  Attack types: {len(detector.faiss_indexer.attack_types)}")
        print(f"  Demo mode: {detector.demo_mode}")
        print(f"  Available analyses: {len(detector.analysis_storage)}")
        
        # Simulate adding some analysis data
        demo_analysis = {
            "threat_level": "HIGH",
            "risk_score": 85,
            "attack_vector": "SYN flood attack",
            "technical_analysis": "High volume TCP SYN packets detected",
            "flow_id": "demo_analysis_001"
        }
        
        detector.analysis_storage["demo_analysis_001"] = demo_analysis
        
        print(f"\n📋 Available analyses: {detector.get_available_analyses()}")
        
        # Demonstrate question queueing
        print(f"\n❓ Demonstrating question queueing...")
        try:
            question_id = detector.ask_question(
                "What makes this attack particularly dangerous?",
                "demo_analysis_001"
            )
            print(f"✅ Question queued with ID: {question_id}")
        except Exception as e:
            print(f"❌ Question queueing failed: {e}")
        
        # Show stats
        stats = detector.get_stats()
        print(f"\n📈 System Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except ImportError:
        print("❌ Detection system not available (missing dependencies)")
        return False
    except Exception as e:
        print(f"❌ Detection system demo failed: {e}")
        return False

def demo_web_interface_features():
    """Demonstrate web interface features."""
    print("\n" + "=" * 60)
    print("🌐 WEB INTERFACE FEATURES")
    print("=" * 60)
    
    print("🎨 Enhanced Web Interface includes:")
    print("  ✅ Real-time flow monitoring")
    print("  ✅ LLM analysis display with RAG indicators")
    print("  ✅ Interactive question-answering interface")
    print("  ✅ Analysis pause/resume controls")
    print("  ✅ RAG-assisted analysis highlighting")
    print("  ✅ Question history and management")
    print("  ✅ Enhanced statistics dashboard")
    
    print("\n🔧 API Endpoints:")
    endpoints = [
        "GET /api/available-analyses - List analyses for questioning",
        "POST /api/ask-question - Submit question about analysis",
        "POST /api/pause-analysis - Pause analysis for questions",
        "POST /api/resume-analysis - Resume normal analysis",
        "GET /api/questions - Get question history"
    ]
    
    for endpoint in endpoints:
        print(f"  • {endpoint}")
    
    print("\n🎯 Key Features:")
    features = [
        "RAG-assisted analysis with visual indicators",
        "Priority question queue system",
        "Interactive analysis selection for questions",
        "Real-time question processing status",
        "Historical analysis search and retrieval",
        "Enhanced GPU monitoring with RAG metrics"
    ]
    
    for feature in features:
        print(f"  🌟 {feature}")

def main():
    """Run the complete feature demonstration."""
    print("🚀 ENHANCED NETWORK IDS FEATURE DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the new question-answering and RAG capabilities")
    print("=" * 80)
    
    # Check system requirements
    print("\n🔍 Checking System Requirements:")
    
    # Check Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  Python: {python_version} {'✅' if sys.version_info >= (3, 8) else '❌'}")
    
    # Check key dependencies
    dependencies = [
        ("torch", "PyTorch for GPU acceleration"),
        ("transformers", "Hugging Face Transformers"),
        ("sentence_transformers", "Sentence Transformers for embeddings"),
        ("faiss", "FAISS for similarity search"),
        ("flask", "Flask web framework"),
        ("flask_socketio", "WebSocket support")
    ]
    
    for dep, desc in dependencies:
        try:
            __import__(dep.replace("-", "_"))
            print(f"  {desc}: ✅")
        except ImportError:
            print(f"  {desc}: ❌ (missing)")
    
    # Run demonstrations
    success_count = 0
    total_demos = 4
    
    if demo_rag_system():
        success_count += 1
    
    if demo_llm_client():
        success_count += 1
    
    if demo_detection_system():
        success_count += 1
    
    demo_web_interface_features()
    success_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 DEMONSTRATION SUMMARY")
    print("=" * 80)
    
    print(f"Completed: {success_count}/{total_demos} demonstrations")
    
    if success_count == total_demos:
        print("🎉 All features demonstrated successfully!")
    else:
        print("⚠️  Some features may require additional setup or dependencies")
    
    print("\n🚀 To use these features:")
    print("1. Ensure all dependencies are installed:")
    print("   pip install -r requirements.txt")
    print("\n2. Run the setup script:")
    print("   python fixed_setup_script.py --setup")
    print("\n3. Start the enhanced web interface:")
    print("   python demo_web_interface.py")
    print("\n4. Access the interface at: http://localhost:5000")
    
    print("\n📚 Additional utilities:")
    print("• python rag_management_utility.py stats - View RAG statistics")
    print("• python rag_management_utility.py search - Interactive RAG search")
    print("• python rag_management_utility.py cleanup - Clean old RAG entries")
    
    print("\n🔍 Key Improvements:")
    improvements = [
        "Question-answering system for detailed analysis insights",
        "RAG integration for historical context in new analyses", 
        "Visual indicators for RAG-assisted analysis",
        "Priority queue system for question processing",
        "Enhanced GPU client with better response logging",
        "Improved web interface with interactive features",
        "RAG management utilities for maintenance"
    ]
    
    for improvement in improvements:
        print(f"  ✨ {improvement}")
    
    print("\n" + "=" * 80)
    print("🎯 The enhanced system is ready for advanced threat analysis!")
    print("=" * 80)

if __name__ == "__main__":
    main()