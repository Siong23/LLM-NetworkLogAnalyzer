from __future__ import annotations

from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
import json
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
import gc
import time
from types import SimpleNamespace

logger = logging.getLogger(__name__)

# Optional tiny RAG for completed analyses
try:
    from analysis_rag_index import AnalysisRAGIndex  # runtime import
except Exception:
    AnalysisRAGIndex = None

if TYPE_CHECKING:
    # forward-ref for type checkers (prevents Pylance error)
    from analysis_rag_index import AnalysisRAGIndex as _AnalysisRAGIndex


class GPUDeepSeekLLMClient:
    """
    Enhanced GPU-optimized DeepSeek LLM client with question-answering capability and improved RAG integration.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: str = "auto",
        max_memory_gb: float = 7.5,
        use_rag: bool = True,
        rag_device: str = "cpu",
        rag_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rag_index_dir: str = "faiss_indices/analysis_rag",
        generation_max_new_tokens: int = 1536,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1
    ):
        self.model_name = model_name
        self.device = device
        self.max_memory_gb = max_memory_gb

        # Enhanced generation parameters
        self.gen_kwargs = {
            "max_new_tokens": generation_max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": None,  # Will be set after tokenizer load
            "eos_token_id": None,  # Will be set after tokenizer load
            "return_full_text": False,
            "clean_up_tokenization_spaces": True
        }

        # Response logging directory
        self.response_log_dir = Path("llm_responses")
        self.response_log_dir.mkdir(exist_ok=True)

        # Model components
        self.model = None
        self.tokenizer = None
        self.generation_pipeline = None

        # RAG
        self.use_rag = use_rag and (AnalysisRAGIndex is not None)
        self.rag: Optional["_AnalysisRAGIndex"] = None
        self.rag_device = rag_device
        self.rag_model_name = rag_model_name
        self.rag_index_dir = rag_index_dir

        # Performance tracking
        self.total_analyses = 0
        self.total_questions = 0
        self.total_inference_time = 0.0

        # Initialization guard
        self._initialized = False

        # Initialize model
        self._initialize_model()
        if self.use_rag:
            try:
                self._init_rag()
            except Exception as e:
                logger.warning(f"RAG disabled (init failed): {e}")
                self.use_rag = False

    # ---------- Lifecycle ----------

    def is_ready(self) -> bool:
        """True if model & pipeline are loaded."""
        return bool(self.generation_pipeline)

    def init(self):
        """Idempotent initialization (so threads can call it)."""
        if not self.is_ready():
            self._initialize_model()
        if self.use_rag and self.rag is None:
            try:
                self._init_rag()
            except Exception as e:
                logger.warning(f"RAG disabled (late init failed): {e}")
                self.use_rag = False
        return self

    def _init_rag(self):
        self.rag = AnalysisRAGIndex(
            index_dir=self.rag_index_dir,
            model_name=self.rag_model_name,
            device=self.rag_device
        )

    def _initialize_model(self):
        if self._initialized:
            return
        logger.info(f"Initializing {self.model_name} with 4-bit quantization...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8
        )
        max_memory = {0: f"{self.max_memory_gb}GB", "cpu": "8GB"}

        # Tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Update generation kwargs with tokenizer info
        self.gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        self.gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        # Model
        logger.info("Loading quantized model to GPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="./offload"
        )

        # Pipeline
        logger.info("Creating generation pipeline...")
        self.generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            return_full_text=False
        )

        self._warmup_model()
        logger.info("Model initialization completed successfully!")
        self._log_memory_usage()
        self._initialized = True

    def _warmup_model(self):
        logger.info("Warming up model...")
        try:
            with torch.inference_mode():
                _ = self.generation_pipeline(
                    "Analyze: network traffic.",
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _log_memory_usage(self):
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

    # ---------- RAG Helpers ----------

    def _build_rag_context(self, flow: Any) -> List[Dict]:
        """Build RAG context with enhanced search query."""
        if not (self.use_rag and self.rag):
            return []
        try:
            # Create comprehensive search query
            attack_type = getattr(flow, 'attack_type', '')
            protocol = getattr(flow, 'protocol', '')
            bytes_count = getattr(flow, 'bytes_count', '')
            packets_count = getattr(flow, 'packets_count', '')
            duration = getattr(flow, 'duration', '')
            
            q = f"{attack_type} attack analysis Protocol={protocol} " \
                f"TotalBytes={bytes_count} TotalPackets={packets_count} " \
                f"Duration={duration} threat assessment"
            
            results = self.rag.search(q, k=3)
            logger.info(f"🔍 RAG found {len(results)} relevant historical analyses")
            return results
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            return []

    def _create_analysis_prompt(self, flow, similar_flows: List[Dict] = None) -> str:
        """Enhanced prompt with improved RAG context integration."""
        rag_ctx = self._build_rag_context(flow)
        rag_block = ""
        rag_used = False
        
        if rag_ctx:
            rag_used = True
            rag_lines = []
            for i, doc in enumerate(rag_ctx):
                snippet = (doc.get("text", "") or "")[:400].replace("\n", " ")
                score = doc.get("score", 0.0)
                rag_lines.append(f"[Analysis {i+1}] (Relevance: {score:.2f}) {snippet}")
            rag_block = "📚 HISTORICAL THREAT ANALYSIS CONTEXT:\n" + "\n".join(rag_lines) + "\n\n"

        prompt = f"""<|system|>
You are an expert cybersecurity analyst. Analyze the network flow and provide a comprehensive threat assessment in valid JSON format.

<|user|>
NETWORK FLOW ANALYSIS REQUEST:

Flow Information:
- Source: {getattr(flow, 'src_ip', '')}:{getattr(flow, 'src_port', '')}
- Destination: {getattr(flow, 'dst_ip', '')}:{getattr(flow, 'dst_port', '')}
- Protocol: {getattr(flow, 'protocol', '')}
- Data Volume: {getattr(flow, 'bytes_count', '')} bytes, {getattr(flow, 'packets_count', '')} packets
- Duration: {getattr(flow, 'duration', '')}s
- TCP Flags: {getattr(flow, 'tcp_flags', '')}
- Detected Attack Type: {getattr(flow, 'attack_type', '')}
- Detection Confidence: {float(getattr(flow, 'confidence', 0.0)):.2f}

{rag_block}"""

        if similar_flows and len(similar_flows) > 0:
            prompt += "Similar Attack Patterns:\n"
            for i, s in enumerate(similar_flows[:2]):
                sim = float(s.get("similarity", 0.0))
                prompt += f"- Pattern {i+1}: similarity={sim:.2f}\n"
            prompt += "\n"

        prompt += f"""CRITICAL: Respond with ONLY a valid JSON object using this EXACT structure:

{{
  "threat_level": "HIGH",
  "risk_score": 85,
  "attack_vector": "Brief description of attack method",
  "potential_impact": "What damage this could cause",
  "immediate_actions": ["Action 1", "Action 2", "Action 3"],
  "investigation_steps": ["Step 1", "Step 2", "Step 3"],
  "prevention_measures": ["Measure 1", "Measure 2", "Measure 3"],
  "technical_analysis": "Detailed technical explanation of the attack pattern and network behavior",
  "rag_assisted": {str(rag_used).lower()}
}}

DO NOT include any text before or after the JSON. Respond with ONLY the JSON object.

<|assistant|>
"""
        return prompt

    def _create_question_prompt(self, original_analysis: Dict, question: str) -> str:
        """Create prompt for asking questions about a specific analysis."""
        prompt = f"""<|system|>
You are an expert cybersecurity analyst. Answer the user's question about a previous threat analysis with detailed, technical information.

<|user|>
ORIGINAL THREAT ANALYSIS:
{json.dumps(original_analysis, indent=2)}

USER QUESTION: {question}

Please provide a detailed, technical answer to the user's question based on the original analysis. Focus on cybersecurity expertise and practical insights. Respond in plain text format (not JSON).

<|assistant|>
"""
        return prompt

    def ingest_analysis_to_rag(self, attack_type: str, flow: Any, analysis_result: Dict):
        """Enhanced RAG ingestion with better metadata."""
        if not (self.use_rag and self.rag):
            return
        
        try:
            # Create comprehensive text for RAG storage
            full_text = f"""Attack Type: {attack_type}
Threat Level: {analysis_result.get('threat_level', 'Unknown')}
Risk Score: {analysis_result.get('risk_score', 0)}/100

Attack Vector: {analysis_result.get('attack_vector', '')}
Potential Impact: {analysis_result.get('potential_impact', '')}

Technical Analysis: {analysis_result.get('technical_analysis', '')}

Immediate Actions: {', '.join(analysis_result.get('immediate_actions', []))}
Investigation Steps: {', '.join(analysis_result.get('investigation_steps', []))}
Prevention Measures: {', '.join(analysis_result.get('prevention_measures', []))}"""

            # Enhanced metadata
            metadata = {
                "attack_type": attack_type,
                "threat_level": analysis_result.get('threat_level', 'Unknown'),
                "risk_score": analysis_result.get('risk_score', 0),
                "timestamp": time.time(),
                "protocol": getattr(flow, "protocol", None),
                "total_packets": getattr(flow, "packets_count", None),
                "total_bytes": getattr(flow, "bytes_count", None),
                "duration": getattr(flow, "duration", None),
                "tcp_flags": getattr(flow, "tcp_flags", None),
                "analysis_source": "gpu_llm_analysis"
            }
            
            self.rag.add_document(full_text, metadata)
            logger.info(f"📚 Ingested {attack_type} analysis into RAG system")
            
        except Exception as e:
            logger.warning(f"RAG ingest failed: {e}")

    # ---------- Response Logging ----------

    def _log_full_response(self, request_type: str, prompt: str, response: str, parsed_result: dict, flow_id: str = None):
        """Enhanced response logging with request type tracking."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_data = {
                "timestamp": timestamp,
                "request_type": request_type,  # "analysis" or "question"
                "flow_id": flow_id,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "full_response": response,
                "parsed_result": parsed_result,
                "model_params": self.gen_kwargs.copy(),
                "total_analyses": self.total_analyses,
                "total_questions": self.total_questions
            }
            
            log_file = self.response_log_dir / f"{request_type}_{timestamp}.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 {request_type.title()} response logged to {log_file.name}")
            
            # Keep only last 100 files to save space
            log_files = sorted(self.response_log_dir.glob("*.json"))
            if len(log_files) > 100:
                for old_file in log_files[:-100]:
                    old_file.unlink()
                    
        except Exception as e:
            logger.warning(f"Failed to log response: {e}")

    # ---------- Inference ----------

    def _generate_analysis(self, prompt: str) -> str:
        """Enhanced generation with better error handling and logging."""
        try:
            logger.info(f"🤖 Starting LLM generation (max_tokens: {self.gen_kwargs['max_new_tokens']})")
            
            with torch.inference_mode():
                outputs = self.generation_pipeline(
                    prompt,
                    **self.gen_kwargs
                )
            
            if outputs and len(outputs) > 0:
                generated_text = (outputs[0]["generated_text"] or "").strip()
                logger.info(f"✅ Generated {len(generated_text)} characters")
                return generated_text
            else:
                logger.error("❌ No output generated from model")
                raise ValueError("No output generated from model")
                
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise

    def answer_question(self, original_analysis: Dict, question: str, flow_id: str = None) -> Dict:
        """Answer a question about a specific analysis."""
        try:
            start_time = time.time()
            
            # Clear GPU cache before analysis
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            prompt = self._create_question_prompt(original_analysis, question)
            
            logger.info(f"❓ Starting question answering for: {question[:50]}...")
            
            # Generate response
            full_text = self._generate_analysis(prompt)
            
            # For questions, we don't parse JSON, just return the text
            result = {
                "question": question,
                "answer": full_text,
                "original_analysis_id": flow_id,
                "response_source": "llm_question_answer",
                "timestamp": datetime.now().isoformat(),
                "inference_time": time.time() - start_time
            }
            
            # Log the Q&A
            self._log_full_response("question", prompt, full_text, result, flow_id)

            # Book-keeping
            self.total_questions += 1
            
            # Clear cache periodically
            if (self.total_analyses + self.total_questions) % 5 == 0:
                torch.cuda.empty_cache()
                logger.info(f"🧹 GPU cache cleared after {self.total_analyses + self.total_questions} total requests")
            
            logger.info(f"✅ Question answered in {result['inference_time']:.2f}s")
            
            return result

        except Exception as e:
            logger.error(f"❌ Question answering failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "error": True,
                "response_source": "error_fallback"
            }

    def analyze_malicious_flow(self, flow, similar_flows: List[Dict] = None) -> Dict:
        try:
            start_time = time.time()
            attack_type = getattr(flow, "attack_type", "unknown")
            
            # Clear GPU cache before analysis
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            prompt = self._create_analysis_prompt(flow, similar_flows)
            
            logger.info(f"🚀 Starting LLM analysis for {attack_type}")
            
            # Generate response
            full_text = self._generate_analysis(prompt)
            
            # Parse response
            analysis = self._parse_llm_response(full_text, flow)
            
            # Check if RAG was used
            rag_assisted = analysis.get('rag_assisted', False)
            if rag_assisted:
                logger.info(f"📚 Analysis was assisted by historical data")
            
            # Auto-ingest into RAG for future use
            self.ingest_analysis_to_rag(attack_type, flow, analysis)
            
            # Log everything for debugging
            flow_id = f"{attack_type}_{int(time.time())}"
            self._log_full_response("analysis", prompt, full_text, analysis, flow_id)

            # Book-keeping
            inference_time = time.time() - start_time
            self.total_analyses += 1
            self.total_inference_time += inference_time
            
            # Clear cache periodically
            if self.total_analyses % 5 == 0:
                torch.cuda.empty_cache()
                logger.info(f"🧹 GPU cache cleared after {self.total_analyses} analyses")
            
            # Add metadata
            analysis["inference_time"] = inference_time
            analysis["model_name"] = self.model_name
            analysis["max_tokens_used"] = self.gen_kwargs["max_new_tokens"]
            analysis["full_response_preview"] = full_text[:200] + "..." if len(full_text) > 200 else full_text
            analysis["full_text"] = full_text[:1000]  # Limit for memory
            analysis["flow_id"] = flow_id
            analysis["rag_context_used"] = len(self._build_rag_context(flow)) > 0
            
            logger.info(f"✅ {attack_type} analysis completed in {inference_time:.2f}s")
            logger.info(f"📊 Response source: {analysis.get('response_source', 'unknown')}")
            logger.info(f"📚 RAG assisted: {rag_assisted}")
            
            return analysis

        except Exception as e:
            logger.error(f"❌ LLM analysis failed for {getattr(flow, 'attack_type', 'unknown')}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fallback = self._create_fallback_analysis(flow)
            fallback["response_source"] = "error_fallback"
            fallback["full_text"] = ""
            fallback["rag_assisted"] = False
            return fallback

    def analyze(self, attack_type: str, flow: Dict[str, Any]) -> Dict:
        """Compatibility method for older interfaces."""
        f = SimpleNamespace(
            timestamp=flow.get("timestamp", datetime.now().isoformat()),
            src_ip=flow.get("src_ip", flow.get("SrcIP", "")),
            dst_ip=flow.get("dst_ip", flow.get("DstIP", "")),
            src_port=flow.get("src_port", flow.get("SrcPort", 0)),
            dst_port=flow.get("dst_port", flow.get("DstPort", 0)),
            protocol=flow.get("protocol", flow.get("Proto", "")),
            bytes_count=flow.get("bytes", flow.get("TotBytes", 0)),
            packets_count=flow.get("packets", flow.get("TotPkts", 0)),
            duration=flow.get("Dur", flow.get("duration", 0.0)),
            tcp_flags=flow.get("State", flow.get("tcp_flags", "")),
            attack_type=attack_type,
            confidence=float(flow.get("confidence", 0.0))
        )
        return self.analyze_malicious_flow(f, similar_flows=None)

    # ---------- Response Parsing ----------

    def _parse_llm_response(self, response: str, flow) -> Dict:
        """Enhanced response parsing with RAG detection."""
        try:
            logger.info(f"🔍 Parsing LLM response of {len(response)} characters")
            
            # Look for JSON in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response[json_start:json_end]
                logger.info(f"🔍 Found JSON block of {len(json_text)} characters")
                
                try:
                    obj = json.loads(json_text)
                    obj = self._validate_analysis(obj)
                    obj['analysis_timestamp'] = datetime.now().isoformat()
                    obj['model_name'] = self.model_name
                    obj['inference_method'] = 'gpu_quantized'
                    obj['response_source'] = 'llm_parsed'
                    obj['full_response_length'] = len(response)
                    
                    # Ensure rag_assisted is boolean
                    obj['rag_assisted'] = bool(obj.get('rag_assisted', False))
                    
                    logger.info("✅ Successfully parsed LLM JSON response")
                    return obj
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ JSON decode failed: {e}")
                    logger.warning(f"Attempted to parse: {json_text[:200]}...")
            
            # If JSON parsing fails, try fallback parsing
            logger.warning("⚠️ JSON parsing failed, using fallback parsing")
            fallback = self._fallback_parse_response(response, flow)
            fallback['response_source'] = 'fallback_parsed'
            fallback['rag_assisted'] = False
            return fallback
            
        except Exception as ex:
            logger.error(f"❌ Parse error: {ex}")
            fallback = self._create_fallback_analysis(flow)
            fallback['response_source'] = 'fallback_analysis'
            fallback['rag_assisted'] = False
            return fallback

    def _validate_analysis(self, a: Dict) -> Dict:
        """Validate and fix analysis structure."""
        defaults = {
            'threat_level': 'HIGH',
            'risk_score': 80,
            'attack_vector': 'Malicious network activity detected',
            'potential_impact': 'Potential network security compromise',
            'immediate_actions': ['Block source IP', 'Monitor traffic patterns'],
            'investigation_steps': ['Analyze logs', 'Check for lateral movement'],
            'prevention_measures': ['Update firewall rules', 'Enhanced monitoring'],
            'technical_analysis': 'Network intrusion attempt detected',
            'rag_assisted': False
        }
        for k, v in defaults.items():
            if k not in a or a[k] in (None, "", []):
                a[k] = v
        try:
            a['risk_score'] = max(0, min(100, int(a['risk_score'])))
        except Exception:
            a['risk_score'] = 80
        if a.get('threat_level') not in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            a['threat_level'] = 'HIGH'
        return a

    def _fallback_parse_response(self, response: str, flow) -> Dict:
        """Fallback parsing when JSON fails."""
        return {
            'threat_level': 'HIGH',
            'risk_score': 75,
            'attack_vector': f'{getattr(flow,"attack_type","")} attack detected',
            'potential_impact': f'Network security threat from {getattr(flow,"src_ip","")}',
            'immediate_actions': [
                f'Block source IP {getattr(flow,"src_ip","")}',
                'Monitor for similar attack patterns',
                'Alert security team immediately'
            ],
            'investigation_steps': [
                'Review full packet capture',
                'Check for indicators of compromise',
                'Analyze network logs for persistence'
            ],
            'prevention_measures': [
                'Update intrusion detection signatures',
                'Implement network segmentation',
                'Enhanced monitoring and alerting'
            ],
            'technical_analysis': (response or 'Analysis generated from fallback method')[:500],
            'parsing_method': 'fallback',
            'rag_assisted': False
        }

    def _create_fallback_analysis(self, flow) -> Dict:
        """Create fallback analysis when everything fails."""
        return {
            'threat_level': 'HIGH',
            'risk_score': 72,
            'attack_vector': f'Detected {getattr(flow,"attack_type","")} attack pattern',
            'potential_impact': f'Potential compromise of {getattr(flow,"dst_ip","")}',
            'immediate_actions': [
                f'Immediately block {getattr(flow,"src_ip","")}',
                f'Monitor traffic to {getattr(flow,"dst_ip","")}',
                'Escalate to security team'
            ],
            'investigation_steps': [
                'Full network traffic analysis',
                'Endpoint security scan',
                'Check for attack persistence'
            ],
            'prevention_measures': [
                'Update security policies',
                'Implement rate limiting',
                'Deploy additional monitoring'
            ],
            'technical_analysis': f'System-generated analysis for {getattr(flow,"attack_type","")} attack',
            'analysis_method': 'fallback_system',
            'timestamp': datetime.now().isoformat(),
            'rag_assisted': False
        }

    # ---------- Performance & Cleanup ----------

    def get_performance_stats(self) -> Dict:
        """Get enhanced performance statistics."""
        avg_time = self.total_inference_time / max(self.total_analyses, 1)
        stats = {
            'total_analyses': self.total_analyses,
            'total_questions': self.total_questions,
            'total_requests': self.total_analyses + self.total_questions,
            'total_inference_time': self.total_inference_time,
            'average_inference_time': avg_time,
            'model_name': self.model_name,
            'max_tokens': self.gen_kwargs['max_new_tokens'],
            'device': str(self.model.device) if self.model else 'unknown',
            'rag_enabled': self.use_rag,
            'rag_index_size': len(getattr(self.rag, 'text_store', [])) if self.rag else 0
        }
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated(0) / 1024**3,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved(0) / 1024**3,
                'gpu_utilization': 'Available'
            })
        return stats

    # COMPATIBILITY METHODS (to ensure the demo works)
    def _maybe_ingest_to_rag(self, attack_type: str, flow: Any, full_text: str):
        """Compatibility method for old code."""
        if full_text:
            dummy_analysis = {"technical_analysis": full_text}
            self.ingest_analysis_to_rag(attack_type, flow, dummy_analysis)

    def cleanup(self):
        """Clean up GPU resources."""
        logger.info("🧹 Cleaning up GPU LLM client...")
        try:
            if hasattr(self, 'generation_pipeline'):
                del self.generation_pipeline
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            self._initialized = False
            logger.info("✅ GPU LLM client cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def shutdown(self):
        self.cleanup()

    def __del__(self):
        self.cleanup()


# Compatibility alias
GPULLMClient = GPUDeepSeekLLMClient

# Test function for demos
def test_gpu_llm_client():
    """Test function for compatibility."""
    try:
        client = GPUDeepSeekLLMClient()
        return client.is_ready()
    except Exception as e:
        logger.error(f"GPU LLM test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    client = GPUDeepSeekLLMClient()
    
    class _Flow:
        timestamp = datetime.now().isoformat()
        src_ip = "192.168.1.20"; dst_ip = "10.0.0.10"
        src_port = 1234; dst_port = 22
        protocol = "TCP"; bytes_count = 1000; packets_count = 10
        duration = 0.1; tcp_flags = "S"; attack_type = "synscan"; confidence = 0.9
    
    print("Testing LLM client...")
    result = client.analyze_malicious_flow(_Flow())
    print(f"Result: {result.get('response_source', 'unknown')} - {result.get('threat_level', 'unknown')}")
    
    # Test question answering
    print("Testing question answering...")
    answer = client.answer_question(result, "What makes this attack dangerous?")
    print(f"Answer: {answer.get('answer', 'No answer')[:100]}...")
    
    client.cleanup()