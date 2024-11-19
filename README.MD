# LightRAG-for-Openwebui

这是一个LightRAG的API模拟器，用于在Openwebui中通过自带的Ollama接口使用LightRAG；通过对话时使用前缀，还可以实现lightrag的模式切换。  
<small style="color:gray;">This is an API simulator for LightRAG, used to utilize LightRAG in Openwebui through the built-in Ollama interface; mode switching of LightRAG can also be achieved by using prefixes during conversations.</small>

## 🎉🎉🎉 声明 🎉🎉🎉  
<small style="color:gray;">## 🎉🎉🎉 Disclaimer 🎉🎉🎉</small>

由于本人没有接触过python和api的各种知识，更不会二次开发openwebui，代码是网上学习过程中使用AI工具编写（包括这个README.md），感谢互联网上大家的开源内容和免费提供的各种ai工具，使得我可以上传我的第一个github小项目。尽管代码实现过程并不优美，但胜在简单易用。如有任何问题或建议，请联系我删除。  
<small style="color:gray;">Since I have no prior knowledge of Python and APIs, nor the ability to redevelop Openwebui, the code was written using AI tools during online learning (including this README.md). Thanks to everyone's open-source content and various free AI tools available on the internet, I was able to upload my first GitHub project. Although the code implementation is not elegant, it is simple and easy to use. If you have any questions or suggestions, please contact me for removal.</small>

## 🚀 快速开始  
<small style="color:gray;">## 🚀 Quick Start</small>

1. 首先你需要正确配置Lightrag的api服务器（[API Server Implementation](https://github.com/HKUDS/LightRAG/blob/main/README.md#api-server-implementation)），并确保可以正常访问。  
   <small style="color:gray;">First, you need to properly configure the Lightrag API server ([API Server Implementation](https://github.com/HKUDS/LightRAG/blob/main/README.md#api-server-implementation)) and ensure it is accessible.</small>

2. 然后你需要正确安装Openwebui（[How to Install](https://github.com/open-webui/open-webui?tab=readme-ov-file#how-to-install-)）。  
   <small style="color:gray;">Then, you need to properly install Openwebui ([How to Install](https://github.com/open-webui/open-webui?tab=readme-ov-file#how-to-install-)).</small>

3. 启动`ApiSimulator.py`（代码块：`python ApiSimulator.py`），转发服务会启动在`localhost:3030`。  
   <small style="color:gray;">Start `ApiSimulator.py` (command: `python ApiSimulator.py`), the forwarding service will start on `localhost:3030`.</small>

4. 打开openwebui，在设置中添加一个Ollama节点，地址设置为`localhost:11434`，然后点击`Verify connection`，此时Openwebui会提示验证成功，`ApiSimulator.py`的运行终端会提示`GET /api/version`和`GET /api/tags`成功。  
   <small style="color:gray;">Open Openwebui, add an Ollama node in the settings, set the address to `localhost:11434`, then click `Verify connection`. Openwebui will indicate a successful verification, and the terminal running `ApiSimulator.py` will show `GET /api/version` and `GET /api/tags` success.</small>

5. 在Openwebui创建新对话，此时会显示`LightragModel:latest`，选择该模型，即可使用LightRAG。  
   <small style="color:gray;">Create a new conversation in Openwebui, `LightragModel:latest` will be displayed, select this model to use LightRAG.</small>

## ⚙️ 配置选项  
<small style="color:gray;">## ⚙️ Configuration Options</small>

在 `ApiSimulator.py` 文件中提供了以下配置选项：  
<small style="color:gray;">The following configuration options are provided in the `ApiSimulator.py` file:</small>

- `OLLAMA_API_URL = "http://localhost:11434"`  Ollama 的 API 地址，如果你使用了 Ollama 的服务，这个地址可以参考你的 Ollama 服务提供的 API 信息来返回版本请求等。  
  <small style="color:gray;">Ollama's API address. If you are using Ollama's service, this address can refer to the API information provided by your Ollama service to return version requests, etc.</small>

- `SERVER_PORT = 3030`  转发服务的端口。  
  <small style="color:gray;">The port for the forwarding service.</small>

- `LIGHTRAG_API_URL = "http://localhost:8020"`  LightRAG API 地址。  
  <small style="color:gray;">LightRAG API address.</small>

- `FAKE_MODEL_NAME = "LightragModel:latest"`  模型名称，可以随意写。  
  <small style="color:gray;">Model name, can be written arbitrarily.</small>

- `FORWARD_MODE = False`  API 请求转发开关，打开之后会直接转发所有请求到 Ollama，用于测试。  
  <small style="color:gray;">API request forwarding switch. When turned on, all requests will be directly forwarded to Ollama for testing.</small>

- `PREFIX_MODE_SWITCH = True`  前缀触发词功能开关，打开之后，对话时输入的内容会根据前缀来判断调用模式，默认为 hybrid。  
  <small style="color:gray;">Prefix trigger word function switch. When turned on, the input content during conversation will determine the call mode based on the prefix, default is hybrid.</small>

- `LOCAL_PREFIX = "_具体地："`  触发 local 模式的前缀。  
  <small style="color:gray;">Prefix to trigger local mode.</small>

- `GLOBAL_PREFIX = "_概括地："`  触发 global 模式的前缀。  
  <small style="color:gray;">Prefix to trigger global mode.</small>

- `NAIVE_PREFIX = "_简单地："`  触发 naive 模式的前缀。  
  <small style="color:gray;">Prefix to trigger naive mode.</small>
