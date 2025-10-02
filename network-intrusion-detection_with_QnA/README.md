# Network Intrusion Detection System with Q&A Capabilities

A comprehensive real-time network intrusion detection system that combines FAISS-based vector similarity search with GPU-accelerated Large Language Model (LLM) analysis and interactive question-answering capabilities.

## 🌟 Features

### Core Detection Capabilities
- **Real-time Network Flow Analysis**: Processes network flows in real-time using FAISS indices
- **Multi-Attack Type Detection**: Supports 8 different attack types from the 5G-NIDD dataset
- **GPU-Accelerated LLM Analysis**: Uses DeepSeek-R1-Distill-Llama-8B for detailed threat analysis
- **Confidence Scoring**: Calibrated confidence scores for classification results

### Advanced Features
- **Interactive Q&A System**: Ask questions about specific threat analyses
- **RAG Integration**: Historical analysis retrieval for enhanced context
- **Real-time Web Interface**: Modern, responsive web dashboard
- **Priority Queue System**: Questions get priority over regular analysis
- **Analysis Pause/Resume**: Control analysis flow for question processing

### Attack Types Supported
- **SYN Flood**: TCP SYN flooding attacks
- **UDP Flood**: UDP flooding attacks  
- **ICMP Flood**: ICMP flooding attacks
- **HTTP Flood**: HTTP-based flooding attacks
- **Slowrate DoS**: Slow-rate denial of service attacks
- **SYN Scan**: TCP SYN port scanning
- **TCP Connect Scan**: Full TCP connection scanning
- **UDP Scan**: UDP port scanning

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Detection Core  │    │   LLM Analysis  │
│                 │    │                  │    │                 │
│ • Real-time UI  │◄──►│ • Flow Generator │◄──►│ • GPU DeepSeek  │
│ • Q&A Interface │    │ • FAISS Indexer │    │ • RAG Enhanced │
│ • Statistics    │    │ • Classification │    │ • Question AI   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   Data & Indices    │
                    │                     │
                    │ • FAISS Indices     │
                    │ • Preprocessors     │
                    │ • Analysis Storage  │
                    └─────────────────────┘
```

## 📋 Requirements

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended 8GB+ VRAM)
- **RAM**: 16GB+ system RAM recommended
- **Python**: 3.8 or higher

### Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
faiss-cpu>=1.7.0
flask>=2.0.0
flask-socketio>=5.0.0
requests>=2.25.0

# GPU LLM Dependencies
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
sentencepiece>=0.1.99
protobuf>=3.20.0
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd network-intrusion-detection_with_QnA/network-intrusion-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup FAISS Indices
```bash
# Build FAISS indices from dataset
python complete_faiss_indexer.py

# Or use the setup script
python fixed_setup_script.py
```

### 3. Start the Web Interface
```bash
python enhanced_web_interface.py
```

### 4. Access the System
Open your browser and navigate to: `http://localhost:5000`

## 🎮 Usage Guide

### Web Interface Controls

#### Starting Detection
1. Choose detection mode:
   - **Demo Mode**: Analyzes one flow per attack type (recommended for testing)
   - **Production Mode**: Analyzes all malicious flows
2. Enable/disable GPU acceleration
3. Click "Start Detection"

#### Monitoring
- **Real-time Flow Monitor**: View classified network flows as they're processed
- **LLM Analysis Panel**: See detailed threat analyses for malicious flows
- **Statistics Dashboard**: Monitor system performance and detection counts

#### Question & Answer System
1. **Select Analysis**: Choose from available threat analyses
2. **Ask Questions**: Enter specific questions about the analysis
3. **Priority Processing**: Questions are processed with higher priority
4. **Pause/Resume**: Control analysis flow during question sessions

### Example Questions
- "What specific indicators suggest this is a SYN flood attack?"
- "How can we prevent this type of attack in the future?"
- "What would be the impact if this attack succeeded?"
- "Are there any similar attack patterns in historical data?"

## 🔧 Configuration

### GPU Configuration
The system automatically detects GPU availability. To force CPU mode:
```python
detector = RealTimeDetectionSystem(use_gpu=False)
```

### Demo vs Production Mode
- **Demo Mode**: Limits LLM analysis to first occurrence of each attack type
- **Production Mode**: Analyzes all detected malicious flows

### Memory Management
- Analysis storage is limited to 100 recent analyses
- Flow history maintains last 200 flows
- Question history keeps last 100 Q&A pairs

## 📊 System Statistics

The system tracks comprehensive statistics:
- **Flow Counts**: Benign, suspicious, malicious flows
- **Analysis Counts**: LLM-analyzed flows
- **Question Counts**: Answered questions
- **Performance Metrics**: Processing rates, queue sizes
- **GPU Metrics**: Memory usage, inference times

## 🗂️ File Structure

```
network-intrusion-detection/
├── enhanced_web_interface.py      # Main web application
├── real_time_detector.py          # Core detection system
├── complete_faiss_indexer.py      # FAISS index management
├── gpu_llm_client.py              # GPU-optimized LLM client
├── analysis_rag_index.py          # RAG system for historical data
├── requirements.txt               # Python dependencies
├── templates/
│   └── index.html                 # Web interface template
├── faiss_indices/                 # FAISS indices and metadata
├── data_split/                    # Dataset splits and indices
└── llm_responses/                 # Stored LLM responses
```

## 🔬 Technical Details

### FAISS Indexing
- **Feature Selection**: 25 most relevant features from 47 total
- **Index Types**: IVFFlat for large datasets, Flat for small ones
- **Calibration**: Per-index distance calibration for confidence scoring
- **Attack Types**: 8 specialized indices for different attack patterns

### LLM Integration
- **Model**: DeepSeek-R1-Distill-Llama-8B (8B parameters)
- **Quantization**: 4-bit quantization for memory efficiency
- **RAG Enhancement**: Historical analysis retrieval for context
- **Question Processing**: Priority queue system for interactive Q&A

### Performance Optimizations
- **GPU Acceleration**: CUDA-optimized inference
- **Memory Management**: Automatic cleanup and limits
- **Queue Systems**: Separate queues for flows, analysis, and questions
- **WebSocket Communication**: Real-time updates to web interface

## 🛠️ Development

### Running in Development Mode
```bash
# Start with debug logging
python enhanced_web_interface.py --debug

# Run detection system standalone
python real_time_detector.py --mode demo --duration 60
```

### Testing Components
```bash
# Test FAISS indexer
python complete_faiss_indexer.py --test

# Test GPU LLM client
python gpu_llm_client.py --test

# Test RAG system
python analysis_rag_index.py --test
```

## 📈 Performance Benchmarks

### Typical Performance (RTX 3080, 16GB RAM)
- **Flow Processing**: 50-100 flows/second
- **FAISS Classification**: <1ms per flow
- **LLM Analysis**: 2-5 seconds per analysis
- **Question Answering**: 1-3 seconds per question
- **Memory Usage**: 6-8GB GPU, 4-6GB RAM

## 🔍 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### FAISS Indices Missing
```bash
# Rebuild indices
python complete_faiss_indexer.py

# Check index directory
ls -la faiss_indices/
```

#### Memory Issues
- Reduce batch sizes in GPU client
- Enable CPU fallback mode
- Increase system swap space

### Logging
The system provides detailed logging. Check console output for:
- Flow processing statistics
- GPU memory usage
- Analysis completion times
- Error messages and warnings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **5G-NIDD Dataset**: Network intrusion detection dataset
- **FAISS**: Facebook AI Similarity Search library
- **DeepSeek**: DeepSeek-R1-Distill-Llama-8B model
- **Hugging Face**: Transformers library and model hosting

## 📞 Support

For questions, issues, or contributions:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information
4. Include system specifications and error logs

---

**Built with ❤️ for network security professionals and researchers**
