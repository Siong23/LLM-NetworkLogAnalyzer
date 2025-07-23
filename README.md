# 🤖 LLM-NetworkLogAnalyzer

**LLM-NetworkLogAnalyzer** is an AI-powered tool that leverages Large Language Models (LLMs) to analyze, summarize, and classify network logs. It aims to assist network administrators and security analysts in making sense of complex and voluminous logs from firewalls, routers, 5G cores, and cloud infrastructure.

---

## 📌 Overview

Network environments generate massive amounts of log data that are difficult to parse manually. LLM-NetworkLogAnalyzer:
- Uses transformer-based models (e.g., GPT, LLaMA, Mistral) to interpret unstructured or semi-structured logs
- Detects anomalies, highlights critical events, and explains patterns in natural language
- Helps automate root cause analysis and event summarization

---

## 🧠 Key Features

- 🔍 **Log Parsing & Normalization**: Converts raw logs into structured formats (JSON, CSV)
- 🧾 **LLM-Powered Summarization**: Produces human-readable summaries of long log traces
- 🚨 **Anomaly Classification**: Uses fine-tuned models to detect and label suspicious events
- 📊 **Insight Generation**: Suggests actions based on recognized patterns
- 🧪 **Prompt Engineering Toolkit**: Supports custom log prompts and tuning

---

## 🖼️ Architecture
```
+--------------------------+
| Network Log Sources      |
| (5G Core, Firewalls,     |
| Routers, Syslog, etc.)   |
+------------+-------------+
             ↓
+--------------------------+
| Log Parser & Formatter   |
| - Regex rules            |
| - Logstash, Fluentd      |
+------------+-------------+
             ↓
+--------------------------+
| LLM Inference Engine     |
| - Prompt templates       |
| - Model backends (OpenAI,|
| HuggingFace, Local)      |
+------------+-------------+
             ↓
+--------------------------+
| Output & Dashboard       |
| - Alerts & Summaries     |
| - Visualizations         |
| - JSON / CSV / UI        |
+--------------------------+
```


---

## 📂 Project Structure
```
LLM-NetworkLogAnalyzer/
├── prompts/ # Prompt templates for different log types
├── parsers/ # Log format converters and regex
├── inference/ # LLM wrapper code (OpenAI API, local models)
├── examples/ # Sample logs and responses
├── tests/ # Unit tests
├── requirements.txt # Python dependencies
└── README.md # This file
```


---

## 🚀 Getting Started

### Prerequisites
```
- Python ≥ 3.9
- OpenAI API key or local LLM (e.g., LLaMA.cpp, Transformers)
- (Optional) Docker for containerized deployment
```
### Installation

```
git clone https://github.com/your-org/LLM-NetworkLogAnalyzer.git
cd LLM-NetworkLogAnalyzer
pip install -r requirements.txt
```
## ⚙️ Usage
Example: Summarize a syslog trace
```
python analyze_log.py --input samples/syslog.log --mode summarize
```
Example: Detect anomalies in 5G core log
```
python analyze_log.py --input samples/5g-core.log --mode classify
```
You can also interactively chat with the logs using:
```
python chat_logs.py --log samples/firewall.log
```

## ✨ Supported Log Formats
- Syslog (RFC 5424)
- 5G Core (Open5GS, Amarisoft)
- Firewall logs (iptables, Palo Alto)
- Web server logs (Apache, NGINX)
- Cloud logs (AWS, GCP, Azure) (WIP)

## 🧪 Example Output
Input (Firewall log):
```
Feb 15 10:31:01 firewall: DROP IN=eth0 OUT= MAC=... SRC=192.168.1.101 DST=10.0.0.5 ...
```
LLM Output (Summary):
Blocked incoming traffic from suspicious internal IP 192.168.1.101 to server 10.0.0.5 on eth0. Possibly scanning or misconfigured host.

## 🔐 Security Considerations
- Avoid sending sensitive logs to external APIs unless anonymized
- Support for on-prem inference with LLaMA or Mistral is available

## 📈 Future Work
- ✅ Fine-tuning on log-specific corpora
- 🌐 Web UI with stream log analysis
- 🔐 Integration with SIEM tools
- 📦 Real-time pipeline (Kafka, ELK, Loki)
