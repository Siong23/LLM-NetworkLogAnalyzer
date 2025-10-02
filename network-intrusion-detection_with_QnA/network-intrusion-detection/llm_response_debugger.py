#!/usr/bin/env python3
"""
LLM Response Debugger - Check if responses are coming from real LLM or fallback
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_llm_responses():
    """Analyze recent LLM responses to check their source and quality."""
    logger.info("=== LLM Response Analysis ===")
    
    response_dir = Path("llm_responses")
    if not response_dir.exists():
        logger.error("No llm_responses directory found!")
        logger.info("Responses will be logged there once the enhanced client runs.")
        return False
    
    # Find recent response files
    response_files = sorted(response_dir.glob("llm_response_*.json"), reverse=True)
    
    if not response_files:
        logger.warning("No response files found!")
        return False
    
    logger.info(f"Found {len(response_files)} response files")
    
    # Analyze last 10 responses
    for i, file_path in enumerate(response_files[:10]):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"\n--- Response {i+1}: {file_path.name} ---")
            logger.info(f"Timestamp: {data.get('timestamp', 'Unknown')}")
            logger.info(f"Response Length: {data.get('response_length', 0)} characters")
            logger.info(f"Model: {data.get('model_name', 'Unknown')}")
            
            # Check response content
            full_response = data.get('full_response', '')
            if len(full_response) > 0:
                logger.info(f"✓ Has full response")
                
                # Check if it looks like JSON
                if '{' in full_response and '}' in full_response:
                    json_start = full_response.find('{')
                    json_end = full_response.rfind('}') + 1
                    json_content = full_response[json_start:json_end]
                    
                    try:
                        parsed = json.loads(json_content)
                        logger.info("✓ Contains valid JSON")
                        logger.info(f"  - Threat Level: {parsed.get('threat_level', 'Unknown')}")
                        logger.info(f"  - Risk Score: {parsed.get('risk_score', 'Unknown')}")
                        logger.info(f"  - Technical Analysis Length: {len(str(parsed.get('technical_analysis', '')))}")
                    except:
                        logger.warning("⚠ JSON parsing failed")
                else:
                    logger.warning("⚠ No JSON structure found")
                
                # Show first 200 chars
                preview = full_response[:200].replace('\n', ' ')
                logger.info(f"Preview: {preview}...")
            else:
                logger.warning("❌ Empty response!")
                
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    return True

def check_current_analysis_results():
    """Check recent analysis results from the system."""
    logger.info("\n=== Current Analysis Results Check ===")
    
    # You can add this to your demo_web_interface.py to see what's actually being processed
    sample_analysis = {
        "threat_level": "HIGH",
        "risk_score": 75,
        "attack_vector": "UDP scan attack detected",
        "potential_impact": "Potential reconnaissance activity",
        "response_source": "unknown"  # This will tell us the source
    }
    
    logger.info("To debug live analysis results, add this to your _analysis_worker:")
    logger.info("""
    # Add after result = self.llm_client.analyze(attack_type, features)
    logger.info(f"LLM Result Source: {result.get('response_source', 'unknown')}")
    logger.info(f"Full Response Length: {result.get('full_response_length', 0)}")
    logger.info(f"Technical Analysis: {result.get('technical_analysis', '')[:100]}...")
    """)

def test_single_llm_call():
    """Test a single LLM call to verify it's working properly."""
    logger.info("\n=== Single LLM Test ===")
    
    try:
        from gpu_llm_client import GPUDeepSeekLLMClient
        from real_time_detector import NetworkFlow
        from datetime import datetime
        
        # Create test client
        client = GPUDeepSeekLLMClient(generation_max_new_tokens=1024)
        
        # Create test flow
        test_flow = NetworkFlow(
            timestamp=datetime.now().isoformat(),
            src_ip="192.168.1.100",
            dst_ip="10.0.0.50", 
            src_port=12345,
            dst_port=22,
            protocol="TCP",
            bytes_count=1500,
            packets_count=25,
            duration=0.5,
            tcp_flags="SYN",
            attack_type="synscan",
            confidence=0.85
        )
        
        logger.info("Running single LLM analysis...")
        result = client.analyze_malicious_flow(test_flow)
        
        logger.info(f"✓ Analysis completed")
        logger.info(f"Response Source: {result.get('response_source', 'unknown')}")
        logger.info(f"Threat Level: {result.get('threat_level', 'unknown')}")
        logger.info(f"Risk Score: {result.get('risk_score', 'unknown')}")
        logger.info(f"Technical Analysis Length: {len(str(result.get('technical_analysis', '')))}")
        
        # Check if response looks like fallback
        tech_analysis = str(result.get('technical_analysis', ''))
        if 'System-generated analysis' in tech_analysis or len(tech_analysis) < 50:
            logger.warning("⚠ This looks like a fallback response!")
        else:
            logger.info("✓ This looks like a real LLM response")
            
        client.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"Single LLM test failed: {e}")
        return False

def main():
    """Run all diagnostic checks."""
    logger.info("🔍 LLM Response Diagnostic Tool")
    logger.info("=" * 50)
    
    # Check logged responses
    analyze_llm_responses()
    
    # Show debugging tips
    check_current_analysis_results()
    
    # Test single call
    test_single_llm_call()
    
    logger.info("\n=== Recommendations ===")
    logger.info("1. Apply the enhanced LLM configuration above")
    logger.info("2. Check llm_responses/ directory for full response logs")
    logger.info("3. Monitor response_source field in analysis results")
    logger.info("4. Increase max_new_tokens to 1024 or higher")
    logger.info("5. Check GPU memory with: nvidia-smi")

if __name__ == "__main__":
    main()