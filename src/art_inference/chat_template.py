from collections.abc import Sequence
import json
import re
from typing import Any

THINKING_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {
    "enable_thinking": False,
    "preserve_thinking": True,
}
TOOL_CALL_ARGUMENTS_AS_MAPPING_ATTR = "_art_tool_call_arguments_as_mapping"
_QWEN_DROP_PRIOR_THINKING = "{%- if loop.index0 > ns.last_query_index %}"
_QWEN_PRESERVE_PRIOR_THINKING = (
    "{%- if (preserve_thinking is defined and preserve_thinking is true) or "
    "(loop.index0 > ns.last_query_index) %}"
)
_GEMMA_DROP_PRIOR_THINKING = (
    "thinking_text and loop.index0 > ns_turn.last_user_idx and "
    "message.get('tool_calls')"
)
_GEMMA_PRESERVE_PRIOR_THINKING = (
    "thinking_text and ((preserve_thinking is defined and preserve_thinking is true) "
    "or (loop.index0 > ns_turn.last_user_idx and message.get('tool_calls')))"
)
_MINIMAX_DROP_PRIOR_THINKING = "reasoning_content and loop.index0 > ns.last_user_index"
_MINIMAX_PRESERVE_PRIOR_THINKING = (
    "reasoning_content and ((preserve_thinking is defined and preserve_thinking is "
    "true) or loop.index0 > ns.last_user_index)"
)
# These operations infer reasoning from arbitrary assistant content and can
# discard everything before the last <think> or between repeated </think> tags.
# Match the operations, not a model revision or the text of a particular answer.
_QWEN_INLINE_STATEMENTS = (
    "if '</think>' in content",
    "set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')",
    "set content = content.split('</think>')[-1].lstrip('\\n')",
    "endif",
)
_QWEN_INLINE_REASONING = re.compile(
    r"\s*".join(
        r"\{%[-+]?\s*" + re.escape(statement) + r"\s*[-+]?%\}"
        for statement in _QWEN_INLINE_STATEMENTS
    )
)


def _without_inline_reasoning_parser(template: str) -> str:
    if "reasoning_content" not in template or "split" not in template:
        return template
    from jinja2 import Environment, TemplateSyntaxError, nodes
    from jinja2.visitor import NodeTransformer

    # Compare parsed operations, not quote/spacing choices or a template hash.
    # Only executable block tokens may be edited; quoted/raw/comment data stays.
    class WithoutWhitespace(NodeTransformer):
        def visit_Output(self, node: nodes.Output, *args: Any, **kwargs: Any):
            if all(
                isinstance(child, nodes.TemplateData) and not child.data.strip()
                for child in node.nodes
            ):
                return None
            return node

    env = Environment()

    def operations(text: str):
        return WithoutWhitespace().visit(env.parse(text)).body

    operation = operations(
        "".join("{% " + statement + " %}" for statement in _QWEN_INLINE_STATEMENTS)
    )
    normalized = re.sub(r"\r\n?", "\n", template)
    offsets = [
        i
        for i, char in enumerate(template)
        if not (char == "\n" and i and template[i - 1] == "\r")
    ] + [len(template)]
    blocks: list[tuple[int, int, int, int]] = []
    cursor = 0
    opening = None
    try:
        for _, kind, value in env.lex(template):
            start = normalized.find(value, cursor)
            if start < 0 or normalized[cursor:start].strip():
                return template  # Lexer normalization could not be source-joined.
            if kind == "block_begin":
                opening = offsets[start], offsets[start + len(value)]
            elif kind == "block_end" and opening is not None:
                # The lexer can include whitespace following a right-trim tag.
                end = start + value.index("%}") + 2
                blocks.append((*opening, offsets[start], offsets[end]))
                opening = None
            cursor = start + len(value)
    except TemplateSyntaxError:
        return template  # Leave invalid templates to their existing renderer.
    edits: dict[tuple[int, int], str] = {}
    for index, (start, _, _, _) in enumerate(blocks):
        selected = blocks[index : index + 4]
        if len(selected) != 4 or "split" not in template[start : selected[-1][3]]:
            continue
        if any(
            template[left[3] : right[0]].strip()
            for left, right in zip(selected, selected[1:])
        ):
            continue
        end = selected[-1][3]
        try:
            if operations(template[start:end]) == operation:
                # The enclosing tags also control unrelated surrounding
                # whitespace. Disable the parser without deleting those tags.
                _, body_start, body_end, first_end = selected[0]
                edits[start, end] = (
                    template[start:body_start]
                    + " if false "
                    + template[body_end:first_end]
                    + template[selected[-1][0] : end]
                )
        except TemplateSyntaxError:
            continue
    if not edits:
        return template

    def apply_edits() -> str:
        result = template
        for (start, end), replacement in sorted(edits.items(), reverse=True):
            result = result[:start] + replacement + result[end:]
        return result

    # Only the content assignment consumed by a recognized parser may lose
    # its trim. Independent preview macros/branches have their own bindings.
    try:
        tree = WithoutWhitespace().visit(env.parse(template))
    except TemplateSyntaxError:
        # Optional binding analysis must not undo independently proved parser
        # edits when the renderer supports extensions absent from this parser.
        return apply_edits()
    assignments = list(tree.find_all(nodes.Assign))
    locations = []
    parsed_assignments = []
    for block in blocks:
        start, body_start, body_end, end = block
        if template[body_start:body_end].split(None, 1)[:1] != ["set"]:
            continue
        try:
            body = env.parse(template[start:end]).body
        except TemplateSyntaxError:
            continue
        if len(body) == 1 and isinstance(body[0], nodes.Assign):
            locations.append(block)
            parsed_assignments.append(body[0])
    if assignments != parsed_assignments:
        return apply_edits()  # Do not guess an assignment's source span.
    # An AST-equivalent block with comments/data between its tags may not be
    # one of the source spans removed above. Join If nodes in source order too.
    conditions = []
    for block in blocks:
        statement = template[block[1] : block[2]].strip()
        keyword = statement.split(None, 1)[:1]
        if keyword not in (["if"], ["elif"]):
            continue
        try:
            parsed = env.parse(
                "{% if " + statement.split(None, 1)[1] + " %}{% endif %}"
            )
        except TemplateSyntaxError:
            return apply_edits()
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], nodes.If):
            return apply_edits()
        conditions.append((block, parsed.body[0].test))
    branches = list(tree.find_all(nodes.If))
    if [node.test for node in branches] != [test for _, test in conditions]:
        return apply_edits()
    edited_parsers = {
        id(node)
        for node, (block, _) in zip(branches, conditions, strict=True)
        if any(start == block[0] for start, _ in edits)
    }
    selected = set()
    shared = set()

    def writes_content(node: nodes.Assign | nodes.AssignBlock) -> bool:
        target = node.target
        return (
            isinstance(target, nodes.Name)
            and target.name == "content"
            or any(n.name == "content" for n in target.find_all(nodes.Name))
        )

    def reads_content(node: nodes.Node) -> bool:
        return (
            isinstance(node, nodes.Name)
            and node.name == "content"
            and node.ctx == "load"
        ) or any(
            n.name == "content" and n.ctx == "load" for n in node.find_all(nodes.Name)
        )

    def assistant_condition(test: nodes.Node) -> bool | None:
        # The rewrite changes assistant content only. Ignore paths proved to
        # handle a different role, but inspect every unknown branch for users
        # of the original trimmed value before the destructive parser.
        if (
            isinstance(test, nodes.Compare)
            and test.expr
            == nodes.Getattr(nodes.Name("message", "load"), "role", "load")
            and len(test.ops) == 1
            and test.ops[0].op in ("eq", "ne")
            and isinstance(test.ops[0].expr, nodes.Const)
        ):
            equal = test.ops[0].expr.value == "assistant"
            return equal if test.ops[0].op == "eq" else not equal
        return None

    def visit(body: Sequence[nodes.Node], binding: nodes.Assign | None = None) -> None:
        for node in body:
            if isinstance(node, nodes.If):
                if id(node) in edited_parsers and node == operation[0]:
                    if binding is not None:
                        selected.add(id(binding))
                    binding = None
                    continue
                if binding is not None and reads_content(node.test):
                    shared.add(id(binding))
                condition = assistant_condition(node.test)
                if condition is not False:
                    visit(node.body, binding)
                if condition is not True:
                    # elif_ is a list of If nodes whose else is stored on the
                    # outer If. Stop following it once a role match is proved.
                    for branch in node.elif_:
                        visit([branch], binding)
                        if assistant_condition(branch.test) is True:
                            break
                    else:
                        visit(node.else_, binding)
                if any(
                    writes_content(n)
                    for n in node.find_all((nodes.Assign, nodes.AssignBlock))
                ):
                    binding = None
            else:
                if binding is not None and reads_content(node):
                    shared.add(id(binding))
                if isinstance(node, nodes.Assign):
                    if writes_content(node):
                        binding = node
                else:
                    # Macro/loop/with/block bodies have independent bindings.
                    for _, value in node.iter_fields():
                        if isinstance(value, list) and all(
                            isinstance(n, nodes.Node) for n in value
                        ):
                            visit(value)
                    if isinstance(node, nodes.AssignBlock) and writes_content(node):
                        binding = None

    visit(tree.body)
    trims = [
        env.parse("{% set content = " + content + " %}").body[0]
        for content in (
            "render_content(message.content, true)|trim",
            "(render_content(message.content, true) if preserve_thinking and message.role == 'assistant' else render_content(message.content, true)|trim)",
        )
    ]
    for node, (start, body_start, body_end, end) in zip(
        assignments, locations, strict=True
    ):
        if id(node) in selected - shared and node in trims:
            edits[start, end] = (
                template[start:body_start]
                + " set content = (render_content(message.content, true) if message.role == 'assistant' else render_content(message.content, true)|trim) "
                + template[body_end:end]
            )
    return apply_edits()


def chat_template_with_preserved_thinking(chat_template: object) -> object:
    """Preserve structured reasoning without interpreting tags in plain content."""
    if isinstance(chat_template, dict):
        return {
            name: chat_template_with_preserved_thinking(template)
            for name, template in chat_template.items()
        }
    if not isinstance(chat_template, str):
        return chat_template
    literal_template = _without_inline_reasoning_parser(chat_template)
    inline_parser_removed = literal_template != chat_template
    chat_template = literal_template
    replacements = (
        (
            _QWEN_DROP_PRIOR_THINKING,
            _QWEN_PRESERVE_PRIOR_THINKING,
            "enable_thinking" in chat_template,
        ),
        (
            _GEMMA_DROP_PRIOR_THINKING,
            _GEMMA_PRESERVE_PRIOR_THINKING,
            True,
        ),
        (
            _MINIMAX_DROP_PRIOR_THINKING,
            _MINIMAX_PRESERVE_PRIOR_THINKING,
            True,
        ),
        (
            "(loop.index0 > ns_turn.last_user_idx) or (preserve_thinking and message.get('tool_calls'))",
            "preserve_thinking or (loop.index0 > ns_turn.last_user_idx)",
            True,
        ),
        *(
            (
                f"message.{field} and not future_final_message.found",
                f"message.{field} and ((preserve_thinking is defined and preserve_thinking is true) or not future_final_message.found)",
                True,
            )
            for field in ("content", "thinking")
        ),
        (
            "{#- CoT is dropped during all previous turns, so we never render it for inference #}",
            "{%- if preserve_thinking is defined and preserve_thinking is true and message.thinking is defined %}<|start|>assistant<|channel|>analysis<|message|>{{ message.thinking }}<|end|>{%- endif %}",
            True,
        ),
        (
            '"<|start|>assistant<|channel|>final<|message|>" + message.content + "<|end|>"',
            '"<|start|>assistant<|channel|>final<|message|>" + message.content + ("<|return|>" if preserve_thinking else "<|end|>")',
            True,
        ),
    )
    for old, new, supported in replacements:
        if supported and chat_template.count(old) == 1:
            chat_template = chat_template.replace(old, new)
    # Kimi 2.5 splits historical turns into a branch that blanks reasoning.
    # Preserve them using its ordinary assistant-turn branch instead.
    if (
        "set hist_msgs = messages[:ns.last_non_tool_call_assistant_msg+1]"
        in chat_template
    ):
        chat_template = (
            chat_template.replace(
                "set hist_msgs = messages[:ns.last_non_tool_call_assistant_msg+1]",
                "set hist_msgs = [] if preserve_thinking else messages[:ns.last_non_tool_call_assistant_msg+1]",
            )
            .replace(
                "set suffix_msgs = messages[ns.last_non_tool_call_assistant_msg+1:]",
                "set suffix_msgs = messages if preserve_thinking else messages[ns.last_non_tool_call_assistant_msg+1:]",
            )
            .replace(
                "{%- if thinking is defined and thinking is false -%}",
                "{%- if thinking is defined and thinking is false and not preserve_thinking -%}",
                1,
            )
        )
    # Qwen's native reasoning parser returns the sampled whitespace. The stock
    # template trims it, then invents a newline before </think>. Preserve the
    # parsed field verbatim; keep the original non-thinking/legacy fallback.
    if (
        "<think>" in chat_template
        and "reasoning_content|trim" in chat_template
        and "if not preserve_thinking or message.reasoning_content" not in chat_template
    ):
        chat_template = chat_template.replace(
            "{%- set reasoning_content = reasoning_content|trim %}",
            "{%- if not preserve_thinking or message.reasoning_content is not string %}"
            "{%- set reasoning_content = reasoning_content|trim %}{%- endif %}",
        )
        chat_template = chat_template.replace(
            "reasoning_content + '\\n</think>\\n\\n'",
            "reasoning_content + ('</think>\\n\\n' if preserve_thinking and message.reasoning_content is string and reasoning_content else '\\n</think>\\n\\n')",
        )
        if (
            not inline_parser_removed
            and "{%- set reasoning_content = reasoning_content|trim %}"
            in literal_template
        ):
            # The recognized parser path already preserved its own input. Do
            # not apply the legacy trim rewrite without its recognized
            # reasoning assignment: an unrelated inline filter can survive
            # parser removal and must not activate this on a second call.
            chat_template = chat_template.replace(
                "set content = render_content(message.content, true)|trim",
                "set content = (render_content(message.content, true) if preserve_thinking and message.role == 'assistant' else render_content(message.content, true)|trim)",
            )
    if "clear_thinking" in chat_template:
        chat_template = chat_template.replace(
            "{{ content.strip() }}",
            "{{ content if not clear_thinking else content.strip() }}",
        ).replace(
            "{%- if content.strip() -%}",
            "{%- if (content if not clear_thinking else content.strip()) -%}",
        )
    if "ns_turn.last_user_idx" in chat_template:
        for value in ("message['content']", "item['text']"):
            chat_template = chat_template.replace(
                f"{{{{- {value} | trim -}}}}",
                f"{{{{- {value} if preserve_thinking and message.role == 'assistant' else {value} | trim -}}}}",
            )
    # Embed preservation defaults so direct tokenizer use and inference engines
    # agree with ART. Passing a value explicitly still takes precedence.
    for name, value in (("preserve_thinking", "true"), ("clear_thinking", "false")):
        default = f"{{%- set {name} = {name} | default({value}) -%}}"
        if name in chat_template and default not in chat_template:
            chat_template = default + chat_template
    return chat_template


def configure_preserved_thinking_chat_template(tokenizer: object) -> object:
    chat_template = getattr(tokenizer, "chat_template", None)
    configured = chat_template_with_preserved_thinking(chat_template)
    if configured != chat_template:
        setattr(tokenizer, "chat_template", configured)
    return tokenizer


def default_chat_template_kwargs_for_template(
    chat_template: object,
) -> dict[str, Any]:
    kwargs = default_preservation_kwargs_for_template(chat_template)
    if not isinstance(chat_template, str):
        return kwargs
    if "enable_thinking" in chat_template:
        kwargs["enable_thinking"] = False
    return kwargs


def default_preservation_kwargs_for_template(chat_template: object) -> dict[str, Any]:
    """History defaults independent of whether the next turn should think."""
    kwargs: dict[str, Any] = {}
    if not isinstance(chat_template, str):
        return kwargs
    if "preserve_thinking" in chat_template:
        kwargs["preserve_thinking"] = True
    if "clear_thinking" in chat_template:
        kwargs["clear_thinking"] = False
    if "deepseek_v4_python_encoder" in chat_template:
        kwargs["drop_thinking"] = False
    return kwargs


def default_chat_template_kwargs_for_tokenizer(tokenizer: object) -> dict[str, Any]:
    return default_chat_template_kwargs_for_template(
        getattr(tokenizer, "chat_template", None)
    )


def merge_chat_template_kwargs(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    return {**(defaults or {}), **(overrides or {})}


def _template_requires_structured_tool_arguments(chat_template: object) -> bool:
    if not isinstance(chat_template, str):
        return False
    arguments_access = (
        r"(?:(?<![.\w])(?:tool_call|tc|function)\s*(?:\.\s*"
        r"(?:function\s*\.\s*)?arguments\b|\[\s*['\"]arguments['\"]\s*\])"
        r"|(?<![.\w])arguments\b)"
    )
    if re.search(rf"{arguments_access}\s*\|\s*items\b", chat_template):
        return True
    if re.search(rf"{arguments_access}\s*\.\s*items\s*\(", chat_template):
        return True
    if re.search(rf"{arguments_access}\s+is\s+mapping\b", chat_template):
        return True
    aliases = re.findall(
        r"{%[-+]?\s*set\s+([A-Za-z_]\w*)\s*=\s*([^%]*)[-+]?%}",
        chat_template,
    )
    return any(
        re.search(arguments_access, expression)
        and re.search(rf"\b{re.escape(alias)}\s*\.\s*items\s*\(", chat_template)
        for alias, expression in aliases
    )


def normalize_tool_call_arguments_for_chat_template(
    messages: list[dict[str, Any]],
    chat_template: object,
    *,
    require_mapping: bool = False,
) -> list[dict[str, Any]]:
    """Give chat templates the structured tool arguments they require.

    Templates that interpolate the raw JSON string must keep string arguments,
    so only templates that iterate structured arguments trigger normalization.
    """
    if not require_mapping and not _template_requires_structured_tool_arguments(
        chat_template
    ):
        return messages
    normalized: list[dict[str, Any]] = []
    for message in messages:
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            normalized.append(message)
            continue
        normalized_calls = []
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            arguments = (
                function.get("arguments") if isinstance(function, dict) else None
            )
            if isinstance(arguments, str):
                assert isinstance(function, dict)
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "tool-call arguments are not valid JSON"
                    ) from error
                if not isinstance(arguments, dict):
                    raise ValueError("tool-call arguments must decode to a JSON object")
                call = {**call, "function": {**function, "arguments": arguments}}
            normalized_calls.append(call)
        normalized.append({**message, "tool_calls": normalized_calls})
    return normalized
