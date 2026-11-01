"""Synthetic tool-call shapes. Round-1 rejected turns are not copied here.

Those ledgers contain model text and stay out of this repository. Each case
below is written to cover one parser variant.
"""

TRANSPORT_FIXTURES = (
    {"name": "strict", "text": '{"tool":"list_files","arguments":{}}', "tool": "list_files", "variant": "strict"},
    {"name": "tool_call", "text": "<tool_call><function=search><parameter=text>needle</parameter></function></tool_call>",
     "tool": "search", "variant": "tool_call"},
    {"name": "tool_call_unclosed", "text": "<tool_call><function=read_file><parameter=path>pkg/mod.py",
     "tool": "read_file", "variant": "tool_call"},
    {"name": "parameter_arguments", "text": '<function=command><parameter=arguments>{"argv":["python3","-m","pytest"]}</parameter></function>',
     "tool": "command", "variant": "parameter_arguments"},
    {"name": "function_tool", "text": "<function=public_tool><parameter=tool>search</parameter><parameter=text>token</parameter></function>",
     "tool": "search", "variant": "function_tool"},
    {"name": "tool_tag", "text": "<tool=read_file><parameter=path>src/app.py</parameter></tool>",
     "tool": "read_file", "variant": "tool_tag"},
    {"name": "dsml", "text": '<｜DSML｜invoke name="search"><｜DSML｜parameter name="text" string="true">needle</｜DSML｜parameter>',
     "tool": "search", "variant": "dsml"},
    {"name": "prose_json", "text": 'I should look this up.\n{"tool":"search","arguments":{"text":"needle"}}',
     "tool": "search", "variant": "prose_json"},
    {"name": "json_fence", "text": '```json\n{"tool":"list_files","arguments":{}}\n```',
     "tool": "list_files", "variant": "json_fence"},
    {"name": "infer_argv", "text": '{"argv":["python3","-c","print(1)"]}', "tool": "command", "variant": "json"},
    {"name": "infer_code", "text": '{"language":"python","code":"print(1)"}', "tool": "execute_code", "variant": "json"},
    {"name": "infer_write", "text": '{"path":"a.py","content":"x=1\\n"}', "tool": "write_file", "variant": "json"},
    {"name": "infer_read", "text": '{"path":"a.py"}', "tool": "read_file", "variant": "json"},
    {"name": "infer_search", "text": '{"text":"needle"}', "tool": "search", "variant": "json"},
    {"name": "extra_arg", "text": '{"tool":"search","arguments":{"text":"needle","unused":1}}',
     "tool": "search", "variant": "strict"},
    {"name": "two_actions", "text": '<tool_call><function=search><parameter=text>one</parameter></function><function=read_file><parameter=path>a.py</parameter></function></tool_call>',
     "tool": "search", "variant": "tool_call", "extra": 1},
)


def replay_fixtures(cases=None):
    """Parse every fixture without a model call. Returns the fraction parsed."""
    from .public_tools import coerce_actions
    from .tool_loop import parse_response
    cases = TRANSPORT_FIXTURES if cases is None else cases
    parsed = 0
    rows = []
    for case in cases:
        text = case["text"]
        tool = None
        variant = "rejected"
        extra = 0
        try:
            action = parse_response(text)
            variant = "strict"
            tool = action.get("tool") or ("final" if "final" in action else None)
        except (ValueError, TypeError, KeyError):
            found = coerce_actions(text)
            if found and found.get("actions"):
                action = found["actions"][0]
                variant = "coerced:" + str(found.get("variant"))
                tool = action.get("tool")
                extra = len(found["actions"]) - 1
        ok = tool == case["tool"] and (case.get("variant") in {None, variant, variant.removeprefix("coerced:")})
        if case.get("extra") is not None and extra != case["extra"]:
            ok = False
        parsed += int(ok)
        rows.append({"name": case["name"], "ok": ok, "tool": tool, "variant": variant, "extra": extra})
    return {"parsed": parsed, "n": len(cases), "rate": parsed / len(cases) if cases else 0.0, "rows": rows}
