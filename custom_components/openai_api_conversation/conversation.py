"""Conversation support for OpenAI API."""

from collections.abc import AsyncGenerator, Callable
import json, ast, re
from typing import Literal
from typing import (
    Any,
    TypedDict,
    Dict,
    List,
    Optional,
    AsyncGenerator,
    cast,
)  # noqa: F811

import openai
from openai._streaming import AsyncStream
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseInputParam,
    ResponseStreamEvent,
    ToolParam,
    WebSearchToolParam,
)
from homeassistant.helpers import (
    entity_registry as er,
    intent,
    template,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.responses.web_search_tool_param import UserLocation
from voluptuous_openapi import convert

from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.components import assist_pipeline, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import OpenAIConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    DEFAULT_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_CITY,
    CONF_WEB_SEARCH_CONTEXT_SIZE,
    CONF_WEB_SEARCH_COUNTRY,
    CONF_WEB_SEARCH_REGION,
    CONF_WEB_SEARCH_TIMEZONE,
    CONF_WEB_SEARCH_USER_LOCATION,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE,
)


class ChatCompletionMessageParam(TypedDict, total=False):
    role: str
    content: str | None
    name: str | None
    tool_calls: list["ChatCompletionMessageToolCallParam"] | None


class Function(TypedDict, total=False):
    name: str
    arguments: str


class ChatCompletionMessageToolCallParam(TypedDict):
    id: str
    type: str
    function: Function


class ChatCompletionToolParam(TypedDict):
    type: str
    function: dict[str, Any]


class ToolCallOutput(TypedDict, total=False):
    call_id: str
    output: str
    id: str


# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 5


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = OpenAIConversationEntity(config_entry)
    async_add_entities([agent])


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> FunctionToolParam:
    """Format tool specification."""
    return FunctionToolParam(
        type="function",
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
        description=tool.description,
        strict=False,
    )


def _format_tool_ext(
    tool: llm.Tool, custom_serializer: Any | None
) -> ChatCompletionToolParam:
    tool_spec = {
        "name": tool.name,
        "description": tool.description or f"用于{tool.name}的工具",
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    return ChatCompletionToolParam(type="function", function=tool_spec)


def create_assistant_message(content, tool_calls):
    tool_call = ChatCompletionMessageToolCallParam(
        id=tool_calls.id,
        type="function",
        function=Function(
            name=tool_calls.tool_name, arguments=json.dumps(tool_calls.tool_args)
        ),
    )
    ret = ChatCompletionMessageParam(
        role="assistant", content=content, tool_calls=[tool_call]
    )
    return ret


def _convert_content_to_param(
    content: conversation.Content,
) -> ResponseInputParam:
    """Convert any native chat message for this agent to the native format."""
    messages: ResponseInputParam = []
    if isinstance(content, conversation.ToolResultContent):
        return [
            FunctionCallOutput(
                role="tool",
                tool_call_id=content.tool_call_id,
                name=content.tool_name,
                content=json.dumps(content.tool_result),
            )
        ]

    if content.content:
        role: Literal["user", "assistant", "system", "developer"] = content.role
        if role == "system":
            # role = "developer"
            role = "system"
        messages.append(
            EasyInputMessageParam(type="message", role=role, content=content.content)
        )

    if isinstance(content, conversation.AssistantContent) and content.tool_calls:
        messages.extend(
            create_assistant_message(
                content.content,
                tool_call,
                # tool_call
            )
            # ResponseFunctionToolCallParam(
            #     type="function",
            #     role="assistant",
            #     name=tool_call.tool_name,
            #     arguments=json.dumps(tool_call.tool_args),
            #     call_id=tool_call.id,
            # )
            for tool_call in content.tool_calls
        )
    return messages


def _fix_invalid_arguments(value: Any) -> Any:
    """Attempt to repair incorrectly formatted json function arguments.

    Small models (for example llama3.1 8B) may produce invalid argument values
    which we attempt to repair here.
    """
    if not isinstance(value, str):
        return value
    value = value.strip()
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            LOGGER.error("OpenAI API error - invalid json arguments: %s", value)
            pass
    return value


def _parse_tool_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ollama tool arguments.

    This function improves tool use quality by fixing common mistakes made by
    small local tool use models. This will repair invalid json arguments and
    omit unnecessary arguments with empty values that will fail intent parsing.
    """
    return {k: _fix_invalid_arguments(v) for k, v in arguments.items() if v}


def is_dict_str(s):
    """判断字符串是否可以安全解析为字典"""
    try:
        result = ast.literal_eval(s)
        return isinstance(result, dict)
    except (SyntaxError, ValueError):
        return False


def repair_dict_str(s):
    """尝试修复不符合字典格式的字符串"""
    # 尝试0: 直接解析原始字符串
    try:
        result = ast.literal_eval(s)
        if isinstance(result, dict):
            return s  # 无需修复
    except:
        pass

    # 步骤1: 修复键未加引号的问题
    repaired = re.sub(r"(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r' "\1":', s)

    # 步骤2: 单引号转双引号 + 转换布尔值和None
    repaired = repaired.replace("'", '"')  # 单引号转双引号
    repaired = re.sub(r"\bTrue\b", "true", repaired)  # True -> true
    repaired = re.sub(r"\bFalse\b", "false", repaired)  # False -> false
    repaired = re.sub(r"\bNone\b", "null", repaired)  # None -> null

    # 步骤3: 修复不完整的键值对
    # 处理以冒号结束但没有值的情况
    repaired = re.sub(r'":\s*([,}])', r'": null\1', repaired)
    repaired = re.sub(r'":\s*$', r'": null', repaired)  # 字符串结尾的情况

    # 步骤4: 修复括号不平衡问题
    open_braces = repaired.count("{")
    close_braces = repaired.count("}")

    # 添加缺失的结束大括号
    if open_braces > close_braces:
        repaired += "}" * (open_braces - close_braces)

    # 步骤5: 尝试修复数组值不完整的情况
    repaired = re.sub(r"\[\s*([,}\]])", r"[ ]\1", repaired)
    repaired = re.sub(r"\[\s*$", r"[ ]", repaired)

    return repaired


def parse_dict_str(s):
    """解析字符串为字典，自动修复格式问题"""
    # 尝试1: 用ast.literal_eval解析原始字符串
    # try:
    #     result = ast.literal_eval(s)
    #     if isinstance(result, dict):
    #         return result
    # except:
    #     pass

    # 尝试2: 修复后再次用ast.literal_eval解析
    repaired = repair_dict_str(s)
    # try:
    #     result = ast.literal_eval(repaired)
    #     if isinstance(result, dict):
    #         return result
    # except:
    #     pass

    # 尝试3: 用json.loads解析修复后的字符串
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        # 尝试4: 作为最后手段，尝试添加外层大括号
        if not repaired.startswith("{") and not repaired.startswith("["):
            try:
                # 尝试作为键值对解析
                return json.loads("{" + repaired + "}")
            except:
                pass
        raise ValueError("无法将字符串解析或修复为字典")


def _fix_invalid_arguments_ext(value: Any) -> Any:
    """Attempt to repair incorrectly formatted json function arguments.

    Small models (for example llama3.1 8B) may produce invalid argument values
    which we attempt to repair here.
    """
    if not isinstance(value, str):
        return value
    value = value.strip()

    if is_dict_str(value):
        LOGGER.info(f"符合dict格式: {value}")
        return json.loads(value)
    else:
        LOGGER.warning(f"不符合dict格式，尝试修复... {value}")
        try:
            # repaired = repair_dict_str(s)
            result = parse_dict_str(value)
            LOGGER.warning(f"解析结果: {result}")
            return result
        except Exception as e:
            LOGGER.error(f"→ 修复失败: {str(e)}")

    # if (value.startswith("[") and value.endswith("]")) or (
    #     value.startswith("{") and value.endswith("}")
    # ):
    #     try:
    #         return json.loads(value)
    #     except json.decoder.JSONDecodeError:
    #         LOGGER.error(
    #             "OpenAI API error - invalid json arguments: %s", value
    #         )
    #         pass
    # return value


async def _transform_stream(
    chat_log: conversation.ChatLog,
    result: AsyncStream[ResponseStreamEvent],
    messages: ResponseInputParam,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform an OpenAI delta stream into HA format."""
    new_msg = True
    tool_calls = []
    try:
        async for chunk in result:
            chunk_ret: conversation.AssistantContentDeltaDict = {}
            if not hasattr(chunk, "choices"):
                raise HomeAssistantError(
                    f"OpenAI API error - no choices in chunk: {chunk}"
                )
            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason
            if new_msg and hasattr(delta, "role") and delta.role:
                chunk_ret["role"] = delta.role
                new_msg = False

            if hasattr(delta, "content") and delta.content:
                chunk_ret["content"] = delta.content
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tool_chunk in delta.tool_calls:
                    if tool_chunk.index >= len(tool_calls):
                        # 创建新的工具调用对象
                        new_tool = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                            "index": tool_chunk.index,
                        }
                        tool_calls.append(new_tool)

                # 获取当前工具对象
                current_tool = tool_calls[tool_chunk.index]

                # 收集工具ID
                if tool_chunk.id:
                    current_tool["id"] = tool_chunk.id

                # 收集函数名称
                if tool_chunk.function and tool_chunk.function.name:
                    current_tool["name"] += tool_chunk.function.name

                # 收集函数参数
                if tool_chunk.function and tool_chunk.function.arguments:
                    current_tool["arguments"] += tool_chunk.function.arguments

            if finish_reason == "tool_calls" and len(tool_calls) > 0:
                LOGGER.info(f"=== ✅ 完成工具chunk获取: {tool_calls} ===")
                # 工具调用完成，修复参数格式
                chunk_ret["tool_calls"] = [
                    llm.ToolInput(
                        id=tool_call["id"],
                        tool_name=tool_call["name"],
                        tool_args=_fix_invalid_arguments(tool_call["arguments"]),
                    )
                    for tool_call in tool_calls
                ]

            if chunk_ret:
                yield chunk_ret

            if finish_reason in ["stop"]:
                LOGGER.debug(f"=== finish_reason: {finish_reason} ===")
                break

    except openai.OpenAIError as err:
        raise HomeAssistantError(f"OpenAI API error: {err}") from err
    except Exception as err:
        raise HomeAssistantError(
            f"Error processing OpenAI API response: {err}"
        ) from err


class OpenAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """OpenAI API conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OpenAIConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="muouandzuozuo",
            model="OpenAI API Conversation",
            # model="qwen_plus",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    def get_exposed_entities(self):
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        exposed_entities = []
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)

            aliases = []
            if entity and entity.aliases:
                aliases = entity.aliases

            exposed_entities.append(
                {
                    "entity_id": entity_id,
                    "domain": state.domain,
                    "name": state.name,
                    "state": self.hass.states.get(entity_id).state,
                    "aliases": aliases,
                }
            )
        return exposed_entities

    def _generate_system_message(
        self, exposed_entities, user_input: conversation.ConversationInput
    ):
        pass

    def _async_generate_custom_prompt(
        self,
        raw_prompt: str,
        exposed_entities,
        user_input: conversation.ConversationInput,
    ) -> str:
        """Generate a prompt for the user."""
        LOGGER.info(f"raw prompt: {raw_prompt}")
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
                "current_device_id": user_input.device_id,
            },
            parse_result=False,
        )

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        options = self.entry.options

        # exposed_entities = self.get_exposed_entities()
        # LOGGER.info(f"=== exposed_entities: {exposed_entities} ===")

        # custom_prompt = self._async_generate_custom_prompt(
        #     options.get(CONF_PROMPT, DEFAULT_PROMPT),
        #     exposed_entities,
        #     user_input,
        # )
        # ee = llm._get_exposed_entities(self.hass,  conversation.DOMAIN, True)

        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[ToolParam] | None = None
        if chat_log.llm_api and hasattr(chat_log.llm_api, "tools"):
            tools = [
                _format_tool_ext(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        if options.get(CONF_WEB_SEARCH):
            web_search = WebSearchToolParam(
                type="web_search_preview",
                search_context_size=options.get(
                    CONF_WEB_SEARCH_CONTEXT_SIZE, RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE
                ),
            )
            if options.get(CONF_WEB_SEARCH_USER_LOCATION):
                web_search["user_location"] = UserLocation(
                    type="approximate",
                    city=options.get(CONF_WEB_SEARCH_CITY, ""),
                    region=options.get(CONF_WEB_SEARCH_REGION, ""),
                    country=options.get(CONF_WEB_SEARCH_COUNTRY, ""),
                    timezone=options.get(CONF_WEB_SEARCH_TIMEZONE, ""),
                )
            if tools is None:
                tools = []
            tools.append(web_search)

        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        LOGGER.info(f"=== Current model: {model} ===")
        messages = [
            m
            for content in chat_log.content
            for m in _convert_content_to_param(content)
        ]

        client = self.entry.runtime_data

        # LOGGER.info(f"Tools available: {tools}")

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            model_args = {
                "model": model,
                "messages": messages,
                "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "user": chat_log.conversation_id,
                "tool_choice": "auto",
                "stream": True,
            }

            LOGGER.debug(f">>> Iteration: {_iteration} Model args: {model_args}")

            if tools:
                model_args["tools"] = tools

            if model.startswith("o"):
                model_args["reasoning"] = {
                    "effort": options.get(
                        CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                    )
                }
            else:
                model_args["store"] = False

            try:
                LOGGER.debug(
                    f"Calling OpenAI API with model: {model}\n, \
                            messages: \n{json.dumps(model_args, ensure_ascii=False, indent=2)}"
                )
                result = await client.chat.completions.create(**model_args)
            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by OpenAI: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except openai.OpenAIError as err:
                LOGGER.error(
                    f">>> Error talking to OpenAI: type: {type(err)}, message: {err}"
                )

                chat_log.async_trace({"error": str(err)})
                # raise HomeAssistantError("Error talking to OpenAI") from err

            async for content in chat_log.async_add_delta_content_stream(
                user_input.agent_id, _transform_stream(chat_log, result, messages)
            ):
                LOGGER.debug(f"Unexpected content type: {type(content)} -> {content}")
                # if not isinstance(content, conversation.AssistantContent):
                messages.extend(_convert_content_to_param(content))

            if not chat_log.unresponded_tool_results:
                break

        intent_response = intent.IntentResponse(language=user_input.language)
        assert type(chat_log.content[-1]) is conversation.AssistantContent
        intent_response.async_set_speech(chat_log.content[-1].content or "")
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)
