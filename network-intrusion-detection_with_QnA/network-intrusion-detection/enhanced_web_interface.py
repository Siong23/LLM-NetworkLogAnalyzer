from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from pathlib import Path
import json
import time
import queue
import threading
from datetime import datetime
from real_time_detector import RealTimeDetectionSystem, NetworkFlow
import logging
import os
import numpy as np
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global detection system instance
detector = None
flow_history = []
analysis_history = []
question_history = []

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class WebSocketDetectionSystem(RealTimeDetectionSystem):
    """Extended detection system that emits updates via WebSocket."""
    
    def __init__(self, demo_mode: bool = True, use_gpu: bool = True):
        super().__init__(demo_mode, use_gpu)
        self.socketio = socketio
        
    def _classification_worker(self):
        """Emit flow updates and enqueue LLM for malicious, and suspicious in demo."""
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

                # enqueue for LLM (malicious always; suspicious too in demo on first-of-type guard inside)
                try:
                    if classification == 'malicious' or (self.demo_mode and classification == 'suspicious' and attack_type):
                        self._handle_malicious_flow(flow)
                        logger.info(f"🚨 Queued {attack_type} for LLM (cls={classification}, conf={confidence:.2f})")
                except Exception as e:
                    logger.warning(f"Could not enqueue for LLM: {e}")

                flow_data = {
                    'id': f"{int(time.time() * 1000)}_{processed_count}",
                    'timestamp': flow.timestamp,
                    'src_ip': flow.src_ip,
                    'dst_ip': flow.dst_ip,
                    'src_port': int(flow.src_port),
                    'dst_port': int(flow.dst_port),
                    'protocol': flow.protocol,
                    'bytes': int(flow.bytes_count),
                    'packets': int(flow.packets_count),
                    'classification': classification,
                    'confidence': float(confidence),  # calibrated 0..1
                    'attack_type': attack_type
                }
                flow_data = convert_numpy_types(flow_data)
                flow_history.append(flow_data)
                if len(flow_history) > 200:
                    flow_history.pop(0)

                stats_data = convert_numpy_types(self.stats.copy())

                try:
                    self.socketio.emit('flow_update', {
                        'flow': flow_data,
                        'stats': stats_data
                    })
                except Exception as e:
                    logger.warning(f"Failed to emit flow_update: {e}")

                if processed_count % 20 == 0 or classification in ('malicious', 'suspicious'):
                    logger.info(f"Flow {processed_count}: {classification} (conf: {confidence:.2f}, type: {attack_type})")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in classification worker: {e}")
                time.sleep(0.1)
    
    def _emit_analysis_result(self, flow: NetworkFlow, result: Dict):
        """Override to emit analysis results to WebSocket."""
        try:
            # FIXED: Build payload that matches frontend expectations
            analysis_payload = {
                "flow": {
                    "id": result.get("flow_id", f"flow_{int(time.time())}"),
                    "src_ip": flow.src_ip,
                    "dst_ip": flow.dst_ip,
                    "protocol": flow.protocol,
                    "attack_type": flow.attack_type,
                    "timestamp": flow.timestamp
                },
                "analysis": {
                    "flow_id": result.get("flow_id"),
                    "threat_level": result.get("threat_level", "HIGH"),
                    "risk_score": result.get("risk_score", 75),
                    "attack_type": flow.attack_type,
                    "attack_vector": result.get("attack_vector", f"{flow.attack_type} attack detected"),
                    "description": result.get("potential_impact", f"Potential {flow.attack_type} attack detected"),
                    "recommendations": result.get("immediate_actions", [
                        f"Block source IP {flow.src_ip}",
                        "Monitor for similar patterns", 
                        "Alert security team"
                    ]),
                    "technical_details": {
                        "confidence": float(getattr(flow, 'confidence', 0.8)),
                        "protocol": flow.protocol,
                        "analysis_time": result.get("inference_time", 0.0),
                        "rag_assisted": result.get("rag_assisted", False),
                        "rag_context_used": result.get("rag_context_used", False)
                    },
                    "technical_analysis": result.get("technical_analysis", f"Detailed analysis of {flow.attack_type} attack pattern"),
                    "analysis_number": self.stats.get("analyzed", 0),
                    "rag_assisted": result.get("rag_assisted", False)
                }
            }

            # Emit to UI with correct event name and structure
            self.socketio.emit("analysis_complete", analysis_payload)
            logger.info(f"Emitted analysis to UI for {flow.attack_type} (RAG: {result.get('rag_assisted', False)})")
            
            # Store in history for API endpoint
            analysis_history.append(analysis_payload["analysis"])
            if len(analysis_history) > 50:
                analysis_history.pop(0)
                
        except Exception as e:
            logger.warning(f"Failed to emit analysis_complete: {e}")
            import traceback
            traceback.print_exc()

    def _emit_question_result(self, result: Dict):
        """Override to emit question results to WebSocket."""
        try:
            question_payload = {
                "question_id": result.get("question_id"),
                "flow_id": result.get("flow_id"),
                "question": result.get("question"),
                "answer": result.get("answer"),
                "timestamp": result.get("timestamp", datetime.now().isoformat()),
                "inference_time": result.get("inference_time", 0.0),
                "error": result.get("error", False)
            }
            
            # Emit to UI
            self.socketio.emit("question_answered", question_payload)
            logger.info(f"Emitted question answer: {result.get('question_id')}")
            
            # Store in history
            question_history.append(question_payload)
            if len(question_history) > 100:
                question_history.pop(0)
                
        except Exception as e:
            logger.warning(f"Failed to emit question result: {e}")

@app.route('/')
def index():
    """Serve the main demo page."""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start the detection system."""
    global detector
    
    data = request.get_json() or {}
    demo_mode = data.get('demo_mode', True)
    use_gpu = data.get('use_gpu', True)
    
    if detector and not detector.stop_event.is_set():
        return jsonify({'error': 'Detection system already running'}), 400
    
    try:
        # Check GPU availability
        gpu_info = None
        if use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    use_gpu = False
                else:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                    gpu_info = {
                        'available': True,
                        'device_name': gpu_name,
                        'memory_gb': float(gpu_memory)  # Ensure it's a Python float
                    }
            except ImportError:
                logger.warning("PyTorch not available")
                use_gpu = False
        
        if not gpu_info:
            gpu_info = {'available': False, 'device_name': None, 'memory_gb': None}
        
        detector = WebSocketDetectionSystem(demo_mode=demo_mode, use_gpu=use_gpu)
        detector.start()
        
        return jsonify({
            'status': 'started', 
            'mode': 'demo' if demo_mode else 'production',
            'gpu_enabled': use_gpu,
            'available_attack_types': detector.faiss_indexer.attack_types,
            'gpu_info': gpu_info
        })
    
    except Exception as e:
        logger.error(f"Failed to start detection system: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop the detection system."""
    global detector
    
    if detector:
        try:
            detector.stop()
            detector = None
            return jsonify({'status': 'stopped'})
        except Exception as e:
            logger.error(f"Error stopping detection system: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'No detection system running'}), 400

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Reset the detection system."""
    global detector, flow_history, analysis_history, question_history
    
    try:
        if detector:
            detector.stop()
        
        # Clear histories
        flow_history.clear()
        analysis_history.clear()
        question_history.clear()
        
        detector = None
        
        socketio.emit('system_reset')
        
        return jsonify({'status': 'reset'})
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get current system statistics."""
    if detector:
        try:
            stats = detector.get_stats()
            return jsonify(convert_numpy_types(stats))
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'benign': 0, 'suspicious': 0, 'malicious': 0, 'analyzed': 0, 'questions_answered': 0})

@app.route('/api/gpu-stats')
def get_gpu_stats():
    """Get GPU performance statistics."""
    if detector and detector.use_gpu and hasattr(detector, 'llm_client'):
        try:
            stats = detector.llm_client.get_performance_stats()
            return jsonify(convert_numpy_types(stats))
        except Exception as e:
            logger.error(f"Error getting GPU stats: {e}")
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'GPU not available or not in use'})

@app.route('/api/flows')
def get_flows():
    """Get recent flows."""
    try:
        flows = flow_history[-20:]  # Return last 20 flows
        return jsonify(convert_numpy_types(flows))
    except Exception as e:
        logger.error(f"Error getting flows: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyses')
def get_analyses():
    """Get recent analyses."""
    try:
        analyses = analysis_history[-10:]  # Return last 10 analyses
        return jsonify(convert_numpy_types(analyses))
    except Exception as e:
        logger.error(f"Error getting analyses: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/available-analyses')
def get_available_analyses():
    """Get list of analyses available for questioning."""
    if detector:
        try:
            analyses = detector.get_available_analyses()
            return jsonify(convert_numpy_types(analyses))
        except Exception as e:
            logger.error(f"Error getting available analyses: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify([])

@app.route('/api/ask-question', methods=['POST'])
def ask_question():
    """Ask a question about a specific analysis."""
    if not detector:
        return jsonify({'error': 'Detection system not running'}), 400
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    question = data.get('question', '').strip()
    flow_id = data.get('flow_id', '').strip()
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    if not flow_id:
        return jsonify({'error': 'Flow ID is required'}), 400
    
    try:
        question_id = detector.ask_question(question, flow_id)
        return jsonify({
            'status': 'queued',
            'question_id': question_id,
            'message': 'Question has been queued for processing'
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        logger.error(f"Error asking question: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/pause-analysis', methods=['POST'])
def pause_analysis():
    """Pause analysis processing to prioritize questions."""
    if not detector:
        return jsonify({'error': 'Detection system not running'}), 400
    
    try:
        detector.pause_analysis()
        return jsonify({'status': 'paused', 'message': 'Analysis paused for question processing'})
    except Exception as e:
        logger.error(f"Error pausing analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/resume-analysis', methods=['POST'])
def resume_analysis():
    """Resume analysis processing."""
    if not detector:
        return jsonify({'error': 'Detection system not running'}), 400
    
    try:
        detector.resume_analysis()
        return jsonify({'status': 'resumed', 'message': 'Analysis processing resumed'})
    except Exception as e:
        logger.error(f"Error resuming analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/questions')
def get_questions():
    """Get recent questions and answers."""
    try:
        questions = question_history[-20:]  # Return last 20 questions
        return jsonify(convert_numpy_types(questions))
    except Exception as e:
        logger.error(f"Error getting questions: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected')
    
    try:
        # Check GPU availability
        gpu_available = False
        gpu_stats = None
        attack_types = []
        
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass
        
        if detector:
            if detector.use_gpu and hasattr(detector, 'llm_client'):
                try:
                    gpu_stats = detector.llm_client.get_performance_stats()
                    gpu_stats = convert_numpy_types(gpu_stats)
                except Exception:
                    pass
            attack_types = detector.faiss_indexer.attack_types
        
        # Convert stats to native Python types
        stats = detector.get_stats() if detector else {'benign': 0, 'suspicious': 0, 'malicious': 0, 'analyzed': 0, 'questions_answered': 0}
        stats = convert_numpy_types(stats)
        
        # Send current state to new client
        emit('initial_state', {
            'flows': convert_numpy_types(flow_history[-20:]),
            'analyses': convert_numpy_types(analysis_history[-10:]),
            'questions': convert_numpy_types(question_history[-10:]),
            'stats': stats,
            'running': detector is not None and not detector.stop_event.is_set() if detector else False,
            'gpu_available': gpu_available,
            'gpu_stats': gpu_stats,
            'attack_types': attack_types,
            'available_analyses': detector.get_available_analyses() if detector else []
        })
    except Exception as e:
        logger.error(f"Error in connect handler: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected')

# Enhanced HTML Template with Question Interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network Intrusion Detection System Demo</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .flow-item { transition: all 0.3s ease; }
        .flow-item.new { animation: slideIn 0.5s ease; }
        @keyframes slideIn {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .rag-indicator {
            background: linear-gradient(45deg, #fbbf24, #f59e0b);
            animation: ragGlow 2s ease-in-out infinite alternate;
        }
        @keyframes ragGlow {
            from { box-shadow: 0 0 5px rgba(251, 191, 36, 0.5); }
            to { box-shadow: 0 0 15px rgba(251, 191, 36, 0.8); }
        }
        .question-input {
            min-height: 80px;
            resize: vertical;
        }
        .analysis-card {
            transition: all 0.3s ease;
        }
        .analysis-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div id="app" class="max-w-7xl mx-auto p-6">
        <!-- Header -->
        <div class="gradient-bg rounded-lg p-6 mb-8 text-white">
            <h1 class="text-3xl font-bold mb-2">
                <i class="fas fa-shield-alt mr-3"></i>
                Network Intrusion Detection System
            </h1>
            <p class="text-blue-100">
                Real-time network flow analysis using FAISS indexing and GPU-accelerated DeepSeek-R1-Distill-Llama-8B with Historical Analysis (RAG)
            </p>
        </div>

        <!-- Controls -->
        <div class="bg-white rounded-lg shadow-sm border p-6 mb-6">
            <div class="flex items-center justify-between mb-4">
                <div class="flex items-center space-x-4">
                    <div class="flex items-center space-x-2">
                        <i class="fas fa-cog text-gray-500"></i>
                        <span class="font-medium">Mode:</span>
                        <select id="modeSelect" class="px-3 py-1 border rounded-md">
                            <option value="true">Demo Mode (1 LLM analysis per attack type)</option>
                            <option value="false">Production Mode (All malicious flows analyzed)</option>
                        </select>
                    </div>
                    <div class="flex items-center space-x-2">
                        <input type="checkbox" id="gpuToggle" checked class="rounded">
                        <label for="gpuToggle" class="text-sm">Use GPU Acceleration</label>
                    </div>
                </div>
                
                <div class="flex items-center space-x-3">
                    <button id="startBtn" class="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition">
                        <i class="fas fa-play"></i>
                        <span>Start Detection</span>
                    </button>
                    
                    <button id="stopBtn" class="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition" disabled>
                        <i class="fas fa-pause"></i>
                        <span>Stop Detection</span>
                    </button>
                    
                    <button id="resetBtn" class="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition">
                        <i class="fas fa-sync-alt"></i>
                        Reset
                    </button>
                </div>
            </div>

            <!-- Statistics -->
            <div class="grid grid-cols-5 gap-4">
                <div class="bg-green-50 border border-green-200 rounded-lg p-4 transition hover:shadow-md">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-green-600 font-medium">Benign Flows</p>
                            <p id="benignCount" class="text-2xl font-bold text-green-800">0</p>
                        </div>
                        <i class="fas fa-check-circle text-green-500 text-2xl"></i>
                    </div>
                </div>
                
                <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 transition hover:shadow-md">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-yellow-600 font-medium">Suspicious Flows</p>
                            <p id="suspiciousCount" class="text-2xl font-bold text-yellow-800">0</p>
                        </div>
                        <i class="fas fa-exclamation-triangle text-yellow-500 text-2xl"></i>
                    </div>
                </div>
                
                <div class="bg-red-50 border border-red-200 rounded-lg p-4 transition hover:shadow-md">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-red-600 font-medium">Malicious Flows</p>
                            <p id="maliciousCount" class="text-2xl font-bold text-red-800">0</p>
                        </div>
                        <i class="fas fa-shield-alt text-red-500 text-2xl"></i>
                    </div>
                </div>
                
                <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 transition hover:shadow-md">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-blue-600 font-medium">LLM Analyzed</p>
                            <p id="analyzedCount" class="text-2xl font-bold text-blue-800">0</p>
                        </div>
                        <i class="fas fa-brain text-blue-500 text-2xl"></i>
                    </div>
                </div>
                
                <div class="bg-purple-50 border border-purple-200 rounded-lg p-4 transition hover:shadow-md">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-purple-600 font-medium">Questions Answered</p>
                            <p id="questionsCount" class="text-2xl font-bold text-purple-800">0</p>
                        </div>
                        <i class="fas fa-question-circle text-purple-500 text-2xl"></i>
                    </div>
                </div>
            </div>
        </div>

        <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
            <!-- Real-time Flow Monitor -->
            <div class="bg-white rounded-lg shadow-sm border">
                <div class="p-4 border-b">
                    <h2 class="text-xl font-semibold text-gray-800">Real-time Flow Monitor</h2>
                    <p class="text-sm text-gray-600">Live network flow classification using FAISS indexing</p>
                </div>
                
                <div class="p-4 max-h-96 overflow-y-auto">
                    <div id="flowContainer" class="space-y-2">
                        <div id="emptyFlowState" class="text-center text-gray-500 py-8">
                            <i class="fas fa-network-wired text-4xl mb-3 opacity-50"></i>
                            <p>No network flows detected yet.</p>
                            <p class="text-sm mt-1">Start the detection to see real-time analysis.</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- LLM Analysis Panel -->
            <div class="bg-white rounded-lg shadow-sm border">
                <div class="p-4 border-b">
                    <h2 class="text-xl font-semibold text-gray-800">DeepSeek LLM Analysis</h2>
                    <p class="text-sm text-gray-600">Detailed threat analysis for malicious flows</p>
                </div>
                
                <div id="analysisContainer" class="p-4 max-h-96 overflow-y-auto">
                    <div id="emptyAnalysisState" class="text-center text-gray-500 py-8">
                        <i class="fas fa-shield-alt text-4xl mb-3 opacity-50"></i>
                        <p>No malicious flows detected yet.</p>
                        <p class="text-sm mt-1">Malicious flows will be automatically sent to the LLM for detailed analysis.</p>
                    </div>
                </div>
            </div>

            <!-- Question Interface -->
            <div class="bg-white rounded-lg shadow-sm border">
                <div class="p-4 border-b">
                    <h2 class="text-xl font-semibold text-gray-800">AI Assistant Q&A</h2>
                    <p class="text-sm text-gray-600">Ask questions about specific threat analyses</p>
                </div>
                
                <div class="p-4">
                    <!-- Question Form -->
                    <div id="questionForm" class="mb-4">
                        <div class="mb-3">
                            <label class="block text-sm font-medium text-gray-700 mb-2">Select Analysis:</label>
                            <select id="analysisSelect" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                                <option value="">Select an analysis to ask about...</option>
                            </select>
                        </div>
                        
                        <div class="mb-3">
                            <label class="block text-sm font-medium text-gray-700 mb-2">Your Question:</label>
                            <textarea 
                                id="questionInput" 
                                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 question-input"
                                placeholder="Ask anything about the selected analysis... Examples:&#10;• What specific indicators suggest this is a SYN flood?&#10;• How can we prevent this type of attack?&#10;• What would be the impact if this attack succeeded?"
                                rows="3"
                            ></textarea>
                        </div>
                        
                        <div class="flex space-x-2">
                            <button id="askQuestionBtn" class="flex-1 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition disabled:bg-gray-400" disabled>
                                <i class="fas fa-question mr-2"></i>
                                Ask Question
                            </button>
                            <button id="pauseBtn" class="bg-yellow-600 text-white px-4 py-2 rounded-md hover:bg-yellow-700 transition">
                                <i class="fas fa-pause mr-2"></i>
                                Pause Analysis
                            </button>
                        </div>
                    </div>
                    
                    <!-- Question History -->
                    <div id="questionContainer" class="max-h-64 overflow-y-auto">
                        <div id="emptyQuestionState" class="text-center text-gray-500 py-8">
                            <i class="fas fa-comments text-4xl mb-3 opacity-50"></i>
                            <p>No questions asked yet.</p>
                            <p class="text-sm mt-1">Select an analysis and ask questions to get detailed insights.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- System Status -->
        <div id="systemStatus" class="mt-6 hidden">
            <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                <div class="flex items-center space-x-2">
                    <div class="w-3 h-3 bg-green-500 rounded-full pulse"></div>
                    <span class="text-green-800 font-medium">
                        System Active - Real-time flow monitoring in progress
                    </span>
                    <span id="modeDisplay" class="text-green-600 text-sm"></span>
                </div>
            </div>
        </div>

        <!-- GPU Status -->
        <div id="gpuStatus" class="mt-4 hidden">
            <div class="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-2">
                        <i class="fas fa-microchip text-purple-600"></i>
                        <span class="text-purple-800 font-medium">GPU Status</span>
                        <span id="gpuName" class="text-purple-600 text-sm"></span>
                    </div>
                    <div class="text-right text-sm text-purple-700">
                        <div>Memory: <span id="gpuMemory">N/A</span></div>
                        <div>Avg Time: <span id="gpuAvgTime">N/A</span></div>
                        <div>RAG Entries: <span id="ragEntries">N/A</span></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // DOM elements
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const resetBtn = document.getElementById('resetBtn');
        const modeSelect = document.getElementById('modeSelect');
        const gpuToggle = document.getElementById('gpuToggle');
        const flowContainer = document.getElementById('flowContainer');
        const analysisContainer = document.getElementById('analysisContainer');
        const questionContainer = document.getElementById('questionContainer');
        const systemStatus = document.getElementById('systemStatus');
        const gpuStatus = document.getElementById('gpuStatus');
        const modeDisplay = document.getElementById('modeDisplay');
        const emptyFlowState = document.getElementById('emptyFlowState');
        const emptyAnalysisState = document.getElementById('emptyAnalysisState');
        const emptyQuestionState = document.getElementById('emptyQuestionState');
        
        // Question interface elements
        const analysisSelect = document.getElementById('analysisSelect');
        const questionInput = document.getElementById('questionInput');
        const askQuestionBtn = document.getElementById('askQuestionBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        
        // State
        let isRunning = false;
        let currentAnalysis = null;
        let flowCount = 0;
        let availableAnalyses = [];
        let isAnalysisPaused = false;
        
        // Event listeners
        startBtn.addEventListener('click', startDetection);
        stopBtn.addEventListener('click', stopDetection);
        resetBtn.addEventListener('click', resetSystem);
        askQuestionBtn.addEventListener('click', askQuestion);
        pauseBtn.addEventListener('click', togglePauseAnalysis);
        
        // Question form validation
        analysisSelect.addEventListener('change', validateQuestionForm);
        questionInput.addEventListener('input', validateQuestionForm);
        
        function validateQuestionForm() {
            const hasAnalysis = analysisSelect.value !== '';
            const hasQuestion = questionInput.value.trim() !== '';
            askQuestionBtn.disabled = !(hasAnalysis && hasQuestion);
        }
        
        // Socket event handlers
        socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        socket.on('initial_state', (data) => {
            console.log('Received initial state:', data);
            updateStats(data.stats);
            updateFlows(data.flows);
            updateAnalyses(data.analyses);
            updateQuestions(data.questions);
            updateAvailableAnalyses(data.available_analyses);
            updateSystemStatus(data.running);
            updateGPUStatus(data.gpu_available, data.gpu_stats);
        });
        
        socket.on('flow_update', (data) => {
            addFlow(data.flow);
            updateStats(data.stats);
        });
        
        socket.on('analysis_complete', (data) => {
            showAnalysisResult(data.analysis);
            updateAvailableAnalyses(); // Refresh available analyses
        });
        
        socket.on('question_answered', (data) => {
            showQuestionResult(data);
            updateStats(); // Refresh stats to show question count
        });
        
        socket.on('gpu_stats', (data) => {
            updateGPUStats(data);
        });
        
        socket.on('system_reset', () => {
            resetUI();
        });
        
        // Functions
        async function startDetection() {
            const demoMode = modeSelect.value === 'true';
            const useGPU = gpuToggle.checked;
            
            try {
                startBtn.disabled = true;
                startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
                
                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        demo_mode: demoMode,
                        use_gpu: useGPU 
                    }),
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    updateSystemStatus(true);
                    modeDisplay.textContent = demoMode ? '(Demo Mode)' : '(Production Mode)';
                    updateGPUStatus(result.gpu_info?.available, null, result.gpu_info);
                } else {
                    alert('Failed to start detection: ' + result.error);
                    startBtn.disabled = false;
                    startBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
                }
            } catch (error) {
                alert('Error starting detection: ' + error.message);
                startBtn.disabled = false;
                startBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
            }
        }
        
        async function stopDetection() {
            try {
                stopBtn.disabled = true;
                stopBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Stopping...';
                
                const response = await fetch('/api/stop', {
                    method: 'POST',
                });
                
                if (response.ok) {
                    updateSystemStatus(false);
                } else {
                    const result = await response.json();
                    alert('Failed to stop detection: ' + result.error);
                }
                
                stopBtn.disabled = false;
                stopBtn.innerHTML = '<i class="fas fa-pause"></i> Stop Detection';
            } catch (error) {
                alert('Error stopping detection: ' + error.message);
                stopBtn.disabled = false;
                stopBtn.innerHTML = '<i class="fas fa-pause"></i> Stop Detection';
            }
        }
        
        async function resetSystem() {
            try {
                const response = await fetch('/api/reset', {
                    method: 'POST',
                });
                
                if (response.ok) {
                    resetUI();
                } else {
                    const result = await response.json();
                    alert('Failed to reset system: ' + result.error);
                }
            } catch (error) {
                alert('Error resetting system: ' + error.message);
            }
        }
        
        async function askQuestion() {
            const flowId = analysisSelect.value;
            const question = questionInput.value.trim();
            
            if (!flowId || !question) {
                alert('Please select an analysis and enter a question.');
                return;
            }
            
            try {
                askQuestionBtn.disabled = true;
                askQuestionBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                
                const response = await fetch('/api/ask-question', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        flow_id: flowId,
                        question: question
                    }),
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    questionInput.value = '';
                    validateQuestionForm();
                    
                    // Show processing state
                    showQuestionProcessing(question);
                } else {
                    alert('Failed to ask question: ' + result.error);
                }
                
                askQuestionBtn.disabled = false;
                askQuestionBtn.innerHTML = '<i class="fas fa-question mr-2"></i> Ask Question';
            } catch (error) {
                alert('Error asking question: ' + error.message);
                askQuestionBtn.disabled = false;
                askQuestionBtn.innerHTML = '<i class="fas fa-question mr-2"></i> Ask Question';
            }
        }
        
        async function togglePauseAnalysis() {
            try {
                const endpoint = isAnalysisPaused ? '/api/resume-analysis' : '/api/pause-analysis';
                const response = await fetch(endpoint, { method: 'POST' });
                
                if (response.ok) {
                    isAnalysisPaused = !isAnalysisPaused;
                    updatePauseButton();
                } else {
                    const result = await response.json();
                    alert('Failed to toggle analysis: ' + result.error);
                }
            } catch (error) {
                alert('Error toggling analysis: ' + error.message);
            }
        }
        
        function updatePauseButton() {
            if (isAnalysisPaused) {
                pauseBtn.innerHTML = '<i class="fas fa-play mr-2"></i> Resume Analysis';
                pauseBtn.className = 'bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition';
            } else {
                pauseBtn.innerHTML = '<i class="fas fa-pause mr-2"></i> Pause Analysis';
                pauseBtn.className = 'bg-yellow-600 text-white px-4 py-2 rounded-md hover:bg-yellow-700 transition';
            }
        }
        
        async function updateAvailableAnalyses(newData = null) {
            if (newData) {
                availableAnalyses = newData;
            } else {
                try {
                    const response = await fetch('/api/available-analyses');
                    if (response.ok) {
                        availableAnalyses = await response.json();
                    }
                } catch (error) {
                    console.error('Error fetching available analyses:', error);
                }
            }
            
            // Update select options
            analysisSelect.innerHTML = '<option value="">Select an analysis to ask about...</option>';
            availableAnalyses.forEach(analysis => {
                const option = document.createElement('option');
                option.value = analysis.flow_id;
                const ragIndicator = analysis.rag_assisted ? '🔍 ' : '';
                option.textContent = `${ragIndicator}${analysis.attack_type} - ${analysis.threat_level} (${new Date(analysis.timestamp).toLocaleTimeString()})`;
                analysisSelect.appendChild(option);
            });
            
            validateQuestionForm();
        }
        
        function updateSystemStatus(running) {
            isRunning = running;
            startBtn.disabled = running;
            stopBtn.disabled = !running;
            modeSelect.disabled = running;
            gpuToggle.disabled = running;
            
            if (running) {
                systemStatus.classList.remove('hidden');
                startBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
            } else {
                systemStatus.classList.add('hidden');
                startBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
                stopBtn.innerHTML = '<i class="fas fa-pause"></i> Stop Detection';
            }
        }
        
        function updateGPUStatus(available, stats, info) {
            if (available && (stats || info)) {
                gpuStatus.classList.remove('hidden');
                
                if (info) {
                    document.getElementById('gpuName').textContent = info.device_name || 'Unknown GPU';
                }
                
                if (stats) {
                    updateGPUStats(stats);
                }
            } else if (!available) {
                gpuStatus.classList.add('hidden');
            }
        }
        
        function updateGPUStats(stats) {
            if (stats.gpu_memory_allocated_gb !== undefined) {
                document.getElementById('gpuMemory').textContent = 
                    stats.gpu_memory_allocated_gb.toFixed(1) + 'GB';
            }
            if (stats.average_inference_time !== undefined) {
                document.getElementById('gpuAvgTime').textContent = 
                    stats.average_inference_time.toFixed(1) + 's';
            }
            if (stats.rag_index_size !== undefined) {
                document.getElementById('ragEntries').textContent = stats.rag_index_size;
            }
        }
        
        function updateStats(stats) {
            if (!stats) return;
            document.getElementById('benignCount').textContent = stats.benign || 0;
            document.getElementById('suspiciousCount').textContent = stats.suspicious || 0;
            document.getElementById('maliciousCount').textContent = stats.malicious || 0;
            document.getElementById('analyzedCount').textContent = stats.analyzed || 0;
            document.getElementById('questionsCount').textContent = stats.questions_answered || 0;
        }
        
        function getClassificationClasses(classification) {
            switch (classification) {
                case 'benign':
                    return 'bg-green-100 border-green-300 text-green-800';
                case 'suspicious':
                    return 'bg-yellow-100 border-yellow-300 text-yellow-800';
                case 'malicious':
                    return 'bg-red-100 border-red-300 text-red-800';
                default:
                    return 'bg-gray-100 border-gray-300 text-gray-800';
            }
        }
        
        function getClassificationIcon(classification) {
            switch (classification) {
                case 'benign':
                    return 'fas fa-check-circle';
                case 'suspicious':
                    return 'fas fa-exclamation-triangle';
                case 'malicious':
                    return 'fas fa-shield-alt';
                default:
                    return 'fas fa-question-circle';
            }
        }
        
        function addFlow(flow) {
            const flowElement = createFlowElement(flow);
            
            // Hide empty state
            if (emptyFlowState) {
                emptyFlowState.style.display = 'none';
            }
            
            // Add new flow at the top
            flowContainer.insertBefore(flowElement, flowContainer.firstChild);
            
            // Keep only last 20 flows
            while (flowContainer.children.length > 21) { // +1 for empty state
                flowContainer.removeChild(flowContainer.lastChild);
            }
            
            flowCount++;
        }
        
        function createFlowElement(flow) {
            const div = document.createElement('div');
            div.className = `flow-item new p-3 rounded-lg border ${getClassificationClasses(flow.classification)}`;
            
            const timestamp = new Date(flow.timestamp).toLocaleTimeString();
            
            div.innerHTML = `
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center space-x-2">
                        <i class="${getClassificationIcon(flow.classification)} w-4 h-4"></i>
                        <span class="font-medium capitalize">${flow.classification}</span>
                        ${flow.attack_type ? `<span class="text-xs px-2 py-1 bg-black bg-opacity-10 rounded">${flow.attack_type}</span>` : ''}
                    </div>
                    <span class="text-xs opacity-75">${timestamp}</span>
                </div>
                
                <div class="text-sm space-y-1">
                    <div class="flex justify-between">
                        <span>${flow.src_ip}:${flow.src_port} → ${flow.dst_ip}:${flow.dst_port}</span>
                        <span class="font-mono">${flow.protocol}</span>
                    </div>
                    <div class="flex justify-between text-xs opacity-75">
                        <span>${flow.bytes} bytes, ${flow.packets} packets</span>
                        <span>Confidence: ${(flow.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
            
            return div;
        }
        
        function updateFlows(flows) {
            // Clear existing flows except empty state
            Array.from(flowContainer.children).forEach(child => {
                if (child.id !== 'emptyFlowState') {
                    flowContainer.removeChild(child);
                }
            });
            
            if (flows && flows.length > 0) {
                emptyFlowState.style.display = 'none';
                flows.forEach(flow => {
                    const flowElement = createFlowElement(flow);
                    flowContainer.appendChild(flowElement);
                });
            } else {
                emptyFlowState.style.display = 'block';
            }
        }
        
        function showAnalysisResult(analysis) {
            emptyAnalysisState.style.display = 'none';
            
            // RAG indicator
            const ragIndicator = analysis.rag_assisted ? 
                '<div class="rag-indicator text-white text-xs px-2 py-1 rounded-full mb-2 inline-block"><i class="fas fa-search mr-1"></i>Historical Analysis Used</div>' : '';
            
            analysisContainer.innerHTML = `
                <div class="space-y-4 analysis-card">
                    ${ragIndicator}
                    <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                        <div class="flex items-center space-x-2 mb-2">
                            <i class="fas fa-shield-alt text-red-600"></i>
                            <span class="font-semibold text-red-800">Threat Detected</span>
                            <span class="px-2 py-1 bg-red-600 text-white text-xs rounded">
                                ${analysis.threat_level}
                            </span>
                        </div>
                        <p class="text-red-700 text-sm mb-2">
                            <strong>Attack Vector:</strong> ${analysis.attack_type}
                        </p>
                        <p class="text-red-600 text-sm">
                            ${analysis.description}
                        </p>
                    </div>

                    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                        <h4 class="font-semibold text-yellow-800 mb-2">Recommended Actions:</h4>
                        <ul class="text-sm text-yellow-700 space-y-1">
                            ${analysis.recommendations.map(rec => `
                                <li class="flex items-start space-x-2">
                                    <span class="text-yellow-600 mt-1">•</span>
                                    <span>${rec}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>

                    <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h4 class="font-semibold text-blue-800 mb-2">Technical Details:</h4>
                        <div class="text-sm text-blue-700 space-y-1">
                            <p><strong>Risk Score:</strong> ${analysis.risk_score}/100</p>
                            <p><strong>Confidence:</strong> ${(analysis.technical_details.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Protocol:</strong> ${analysis.technical_details.protocol}</p>
                            <p><strong>Analysis Time:</strong> ${analysis.technical_details.analysis_time.toFixed(2)}s</p>
                            <p><strong>Analysis #:</strong> ${analysis.analysis_number}</p>
                            <p><strong>Flow ID:</strong> ${analysis.flow_id}</p>
                            ${analysis.rag_assisted ? '<p><strong>🔍 Enhanced by Historical Data</strong></p>' : ''}
                        </div>
                    </div>

                    <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <h4 class="font-semibold text-gray-800 mb-2">Technical Analysis:</h4>
                        <p class="text-sm text-gray-700">${analysis.technical_analysis}</p>
                    </div>
                </div>
            `;
        }
        
        function showQuestionProcessing(question) {
            emptyQuestionState.style.display = 'none';
            
            const questionElement = document.createElement('div');
            questionElement.className = 'border border-blue-200 rounded-lg p-4 mb-4 bg-blue-50';
            questionElement.innerHTML = `
                <div class="text-center">
                    <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                    <p class="text-blue-700 font-medium">Processing question...</p>
                    <p class="text-blue-600 text-sm mt-1">${question}</p>
                </div>
            `;
            
            questionContainer.insertBefore(questionElement, questionContainer.firstChild);
        }
        
        function showQuestionResult(data) {
            emptyQuestionState.style.display = 'none';
            
            // Remove any processing state
            const processingElements = questionContainer.querySelectorAll('.bg-blue-50');
            processingElements.forEach(el => el.remove());
            
            const questionElement = document.createElement('div');
            questionElement.className = `border rounded-lg p-4 mb-4 ${data.error ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}`;
            
            const timestamp = new Date(data.timestamp).toLocaleTimeString();
            
            questionElement.innerHTML = `
                <div class="space-y-3">
                    <div class="flex items-start justify-between">
                        <div class="flex items-center space-x-2">
                            <i class="fas fa-question-circle ${data.error ? 'text-red-600' : 'text-green-600'}"></i>
                            <span class="font-medium ${data.error ? 'text-red-800' : 'text-green-800'}">Question</span>
                            <span class="text-xs px-2 py-1 ${data.error ? 'bg-red-200 text-red-700' : 'bg-green-200 text-green-700'} rounded">
                                ${data.error ? 'Error' : 'Answered'}
                            </span>
                        </div>
                        <span class="text-xs ${data.error ? 'text-red-600' : 'text-green-600'}">${timestamp}</span>
                    </div>
                    
                    <div class="bg-white bg-opacity-50 rounded p-3">
                        <p class="text-sm font-medium ${data.error ? 'text-red-700' : 'text-green-700'} mb-2">Q: ${data.question}</p>
                        <p class="text-sm ${data.error ? 'text-red-600' : 'text-green-600'}">${data.answer}</p>
                    </div>
                    
                    ${!data.error && data.inference_time ? `
                        <div class="text-xs ${data.error ? 'text-red-500' : 'text-green-500'}">
                            Response time: ${data.inference_time.toFixed(2)}s
                        </div>
                    ` : ''}
                </div>
            `;
            
            questionContainer.insertBefore(questionElement, questionContainer.firstChild);
            
            // Keep only last 10 questions
            while (questionContainer.children.length > 11) { // +1 for empty state
                questionContainer.removeChild(questionContainer.lastChild);
            }
        }
        
        function updateAnalyses(analyses) {
            if (analyses && analyses.length > 0) {
                showAnalysisResult(analyses[analyses.length - 1]);
            } else {
                emptyAnalysisState.style.display = 'block';
            }
        }
        
        function updateQuestions(questions) {
            if (questions && questions.length > 0) {
                emptyQuestionState.style.display = 'none';
                questions.slice(-5).reverse().forEach(question => {
                    showQuestionResult(question);
                });
            } else {
                emptyQuestionState.style.display = 'block';
            }
        }
        
        function resetUI() {
            // Reset flows
            Array.from(flowContainer.children).forEach(child => {
                if (child.id !== 'emptyFlowState') {
                    flowContainer.removeChild(child);
                }
            });
            emptyFlowState.style.display = 'block';
            
            // Reset analysis
            analysisContainer.innerHTML = '';
            analysisContainer.appendChild(emptyAnalysisState);
            emptyAnalysisState.style.display = 'block';
            
            // Reset questions
            Array.from(questionContainer.children).forEach(child => {
                if (child.id !== 'emptyQuestionState') {
                    questionContainer.removeChild(child);
                }
            });
            emptyQuestionState.style.display = 'block';
            
            // Reset form
            analysisSelect.innerHTML = '<option value="">Select an analysis to ask about...</option>';
            questionInput.value = '';
            validateQuestionForm();
            
            // Reset stats
            updateStats({benign: 0, suspicious: 0, malicious: 0, analyzed: 0, questions_answered: 0});
            
            // Reset status
            updateSystemStatus(false);
            gpuStatus.classList.add('hidden');
            
            flowCount = 0;
            availableAnalyses = [];
            isAnalysisPaused = false;
            updatePauseButton();
        }
        
        // Initialize UI
        resetUI();
    </script>
</body>
</html>
'''

# Create templates directory and save HTML template
def create_templates():
    """Create templates directory and HTML file."""
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        logger.info(f"Created templates directory: {templates_dir}")

    template_path = os.path.join(templates_dir, 'index.html')
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)
    logger.info(f"Updated HTML template: {template_path}")

if __name__ == '__main__':
    logger.info("🛡️  Starting Enhanced Network Intrusion Detection Demo Web Interface")
    
    # Create templates directory and HTML file
    create_templates()
    
    # Check dependencies
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("No GPU detected - CPU fallback will be used")
    except ImportError:
        logger.warning("PyTorch not installed - GPU acceleration will not be available")
    
    # Check if FAISS indices exist
    indices_path = Path("faiss_indices")
    if not indices_path.exists() or not (indices_path / "preprocessing.pkl").exists():
        logger.warning("FAISS indices not found!")
        logger.warning("Please run: python complete_faiss_indexer.py")
        logger.warning("Or use: python setup_and_run.py --setup")
    
    logger.info("Enhanced Features:")
    logger.info("1. ✓ Question-Answering System for LLM responses")
    logger.info("2. ✓ RAG Integration with historical analysis")
    logger.info("3. ✓ Visual indicators for RAG-assisted analysis")
    logger.info("4. ✓ Priority queue system for questions")
    logger.info("5. ✓ Analysis pause/resume functionality")
    logger.info("")
    logger.info("Web interface will be available at: http://localhost:5000")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Cleanup
        if detector:
            try:
                detector.stop()
            except:
                pass