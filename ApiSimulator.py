import requests
from flask import Flask, request, jsonify, Response
import logging

app = Flask(__name__)

# 配置选项 / Configuration options
OLLAMA_API_URL = "http://localhost:11434"  # Ollama的API地址，如果你使用了Ollama的服务，这个地址可以参考你的ollama服务提供的api信息来返回版本请求等。/ Ollama API address, if you use Ollama service, this address can refer to the API information provided by your Ollama service to return version requests, etc.
SERVER_PORT = 3030  # 转发服务的端口 / Port for forwarding service
LIGHTRAG_API_URL = "http://localhost:8020"  # Lightrag API地址 / Lightrag API address
FAKE_MODEL_NAME = "LightragModel:latest"  # 模型名称，可以随意写 / Model name, can be arbitrary
FORWARD_MODE = False  # API请求转发开关，打开之后会直接转发所有请求到Ollama，用于测试。/ API request forwarding switch, when turned on, all requests will be directly forwarded to Ollama for testing.
PREFIX_MODE_SWITCH = True  # 前缀触发词功能开关，打开之后，对话时输入的内容会根据前缀来判断调用模式，默认为hybrid。/ Prefix trigger word function switch, when turned on, the input content during conversation will determine the call mode based on the prefix, default is hybrid.
LOCAL_PREFIX = "_具体地："  # 触发local模式的前缀 / Prefix to trigger local mode
GLOBAL_PREFIX = "_概括地："  # 触发global模式的前缀 / Prefix to trigger global mode
NAIVE_PREFIX = "_简单地："  # 触发naive模式的前缀 / Prefix to trigger naive mode

logging.basicConfig(level=logging.DEBUG)

def call_lightrag_api(query, mode="hybrid", only_need_context=False):
    """调用Lightrag API并返回响应 / Call Lightrag API and return response"""
    headers = {"Content-Type": "application/json"}
    lightrag_request = {"query": query, "mode": mode, "only_need_context": only_need_context}
    response = requests.post(f"{LIGHTRAG_API_URL}/query", headers=headers, json=lightrag_request)
    return response.json() if response.status_code == 200 else None

def determine_mode_and_strip_prefix(content):
    """根据内容前缀确定调用模式并去掉前缀 / Determine call mode based on content prefix and remove prefix"""
    if PREFIX_MODE_SWITCH:
        if content.startswith(LOCAL_PREFIX):
            return "local", content[len(LOCAL_PREFIX):]
        elif content.startswith(GLOBAL_PREFIX):
            return "global", content[len(GLOBAL_PREFIX):]
        elif content.startswith(NAIVE_PREFIX):
            return "naive", content[len(NAIVE_PREFIX):]
    return "hybrid", content

@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天请求 / Handle chat requests"""
    data = request.get_json()
    logging.debug(f"Received data: {data}")

    if not data or "messages" not in data or not data["messages"] or "model" not in data:
        logging.error("Invalid request format")
        return jsonify({"error": "Invalid request format"}), 400

    is_streaming = data.get("stream", True)
    content = data["messages"][-1]["content"]
    mode, stripped_content = determine_mode_and_strip_prefix(content)

    if FORWARD_MODE:
        return forward_request_to_ollama(data)

    lightrag_response = call_lightrag_api(stripped_content, mode=mode)
    if lightrag_response and "data" in lightrag_response:
        ollama_response = format_response(data["model"], lightrag_response["data"])
        return stream_response(ollama_response) if is_streaming else jsonify(ollama_response)
    else:
        logging.error("Lightrag API response does not contain 'data'")
        return jsonify({"error": "Failed to get valid response from Lightrag API"}), 500

def forward_request_to_ollama(data):
    """转发请求到OLLAMA API / Forward request to OLLAMA API"""
    try:
        with requests.post(f"{OLLAMA_API_URL}/api/chat", json=data, stream=True) as response:
            if response.status_code != 200:
                logging.error("Failed to forward request to OLLAMA API")
                return jsonify({"error": "Failed to forward request to OLLAMA API"}), response.status_code

            return Response(response.iter_content(chunk_size=8192), content_type=response.headers.get('Content-Type'))
    except requests.RequestException as e:
        logging.error(f"Error forwarding request: {e}")
        return jsonify({"error": f"Failed to forward request to OLLAMA API: {str(e)}"}), 500

def format_response(model, content):
    """格式化响应为Ollama格式 / Format response to Ollama format"""
    return {
        "model": model,
        "created_at": "2023-12-12T14:13:43.416799Z",
        "message": {"role": "assistant", "content": content, "images": None},
        "done": True,
        "total_duration": 5191566416,
        "load_duration": 2154458,
        "prompt_eval_count": 26,
        "prompt_eval_duration": 383809000,
        "eval_count": 298,
        "eval_duration": 4799921000
    }

def stream_response(ollama_response):
    """生成流式响应 / Generate streaming response"""
    def generate_stream():
        with app.app_context():
            yield jsonify({
                "model": ollama_response["model"],
                "created_at": ollama_response["created_at"],
                "message": {"role": "assistant", "content": ollama_response["message"]["content"], "images": None},
                "done": False
            }).data
            yield jsonify(ollama_response).data

    return Response(generate_stream(), content_type='application/json')

@app.route('/api/version', methods=['GET'])
def get_version():
    """获取API版本信息 / Get API version information"""
    if FORWARD_MODE:
        response = requests.get(f"{OLLAMA_API_URL}/api/version")
        return (response.content, response.status_code, response.headers.items())

    version_info = {"version": "0.3.35", "description": "A fake API server for OpenWebUI"}
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/version")
        if response.status_code == 200:
            version_info = response.json()
    except requests.RequestException:
        pass

    return jsonify(version_info)

@app.route('/api/tags', methods=['GET'])
def get_tags():
    """获取本地模型信息 / Get local model information"""
    if FORWARD_MODE:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        return (response.content, response.status_code, response.headers.items())

    fake_model_info = {
        "name": FAKE_MODEL_NAME,
        "model": FAKE_MODEL_NAME,
        "modified_at": "2023-11-04T14:56:49.277302595-07:00",
        "size": 7365960935,
        "digest": "9f438cb9cd581fc025612d27f7c1a6669ff83a8bb0ed86c94fcf4c5440555697",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "13B",
            "quantization_level": "Q4_0"
        }
    }
    return jsonify({"models": [fake_model_info]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=SERVER_PORT)