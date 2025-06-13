
<div align="center">

### Openai API Home Assistant 🏡


![GitHub Version](https://img.shields.io/github/v/release/muou55555/openai_api_conversation) 
![GitHub Issues](https://img.shields.io/github/issues/muou55555/openai_api_conversation) 
![GitHub Forks](https://img.shields.io/github/forks/muou55555/openai_api_conversation?style=social) 
![GitHub Stars](https://img.shields.io/github/stars/muou55555/openai_api_conversation?style=social)
</div>
<br>

# OpenAI API Conversation Integration


This integration enables natural language interactions with OpenAI models within Home Assistant, providing a seamless conversational interface for users.

## Integration Overview
### Core Features
- [x] Support for multiple OpenAI api models. eg: qwen、doubao、deepseek、gpt-4o, etc.
- [x] Conversation history tracking
- [x] Configure model parameters (temperature, max_tokens, etc.)
- [x] Understands conversation history and device states for coherent interaction
- [x] Seamless integration with Home Assistant entities、services and srcripts etc.
- [x] Supports streaming conversations
- [x] Supports custom prompts and context
- [x] Supports add multiple models and assistants 
- [x] Supports customize the MCP server API
- [x] Supports web search 

### Planned Features
- [ ] Supports image generation services
- [ ] Supports custom intents and triggers
- [ ] Supports automated scenarios and workflows

---

## Installation

### Manual Installation:
1. Download the component files and place them in `custom_components/openai_api_conversation/`
2. Restart Home Assistant

---

## Configuration

### Step 1: Obtain API Key
   - Visit [volcengine](https://www.volcengine.com//api-keys) or [Aliyun](https://bailian.console.aliyun.com/) to get your API key.
   - Get web url, examples: 
     -   volcengine: ```https://ark.cn-beijing.volces.com/api/v3```
     -   qwen: ```https://dashscope.aliyuncs.com/compatible-mode/v1```

### Step 2: Configure via UI
   1. Go to **Settings** → **Devices & Services**
   2. Click **Add integration** and select "OpenAI API Conversation"
   3. Enter your Base_url and API key and save

---

## Extended Usage Examples

### Service Call Example
Use this in automations or scripts to send messages:
```yaml
service: openai_api_conversation.generate_content
data:
  message: "What's the current temperature?"
  model: "qwen-turbo"
  temperature: 0.7
```

---

## Dependencies
- Python requirements: `openai>=1.76.2` (automatically handled via HACS)

---

## Notes
- Ensure internet connectivity for API requests
- API key security: Never expose your key in templates/automations
- The extended_openai_conversation integration needs to be disabled.
- If you want to use the internet connectivity feature, try not to use the internet search function under zhipuai, as conflicts may occur.
