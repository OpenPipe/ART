from IPython.display import display, HTML  # Add import
import html  # Add import for escaping

from art_e.data.query_iterators import SyntheticQuery
from art_e.email_search_tools import read_email


def display_run_html(row: dict, scenarios: list[SyntheticQuery]):
    """Displays the details of a single run as HTML in a Jupyter environment."""
    scenario_id = int(row["metadata_scenario_id"])
    scenario = next((s for s in scenarios if s.id == scenario_id), None)

    html_parts = []

    if scenario is None:
        html_parts.append(f"<p><b>Scenario {scenario_id} not found</b></p>")
        display(HTML("".join(html_parts)))
        return

    email_html = "<p><i>No email ID associated with scenario.</i></p>"
    if scenario.message_ids and scenario.message_ids[0] is not None:
        try:
            message = read_email(scenario.message_ids[0])
            if message is None:
                email_html = f"<p><i>Email {scenario.message_ids[0]} not found for Scenario {scenario_id}</i></p>"
            else:
                email_subject_id = html.escape(
                    f"{message.subject} ({message.message_id})"
                )
                # Ensure body is a string before escaping
                email_body_str = message.body if message.body is not None else ""
                email_body = html.escape(email_body_str).replace("\n", "<br>")
                email_html = f"""
                <h4>Source Email: {email_subject_id}</h4>
                <pre style="background-color: #f5f5f5; padding: 10px; border: 1px solid #ddd; white-space: pre-wrap; word-wrap: break-word;">{email_body}</pre>
                """
        except Exception as e:
            error_message = html.escape(
                f"Error reading email {scenario.message_ids[0]} for Scenario {scenario_id}: {e}"
            )
            email_html = f'<p style="color: red;"><i>{error_message}</i></p>'
    else:
        email_html = f"<p><i>No message ID found for Scenario {scenario_id}</i></p>"

    # Header
    html_parts.append(
        '<div style="border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; background-color: #f9f9f9;">'
    )
    html_parts.append(f"<h2>Run Details (Model: {html.escape(str(row['model']))})</h2>")

    # Scenario Info
    html_parts.append(f"<p><b>Scenario ID:</b> {scenario.id}</p>")
    html_parts.append(f"<p><b>Question:</b> {html.escape(scenario.question)}</p>")
    html_parts.append(f"<p><b>Expected Answer:</b> {html.escape(scenario.answer)}</p>")

    # Email Content
    html_parts.append(email_html)

    # Messages / Chat History
    html_parts.append("<h3>Chat History:</h3>")
    if row["messages"]:
        html_parts.append('<div style="margin-left: 20px;">')

        # Define colors for different roles
        role_colors = {
            "user": "#e1f5fe",  # Light blue
            "assistant": "#e8f5e9",  # Light green
            "system": "#f3e5f5",  # Light purple
            "tool": "#fffde7",  # Light yellow
        }

        for message in row["messages"]:
            role = message["role"]  # No need to escape if just used for key/display
            content = message["content"]
            # Attempt to parse tool_calls if it's a string representation of a list of dicts
            tool_calls_data = None
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, str):  # Check if it's a string first
                try:
                    # This assumes the string is a valid Python literal representation
                    import ast

                    tool_calls_data = ast.literal_eval(raw_tool_calls)
                except (ValueError, SyntaxError):
                    # Keep raw string if parsing fails
                    tool_calls_data = raw_tool_calls
            elif isinstance(raw_tool_calls, list):  # Handle if already a list
                tool_calls_data = raw_tool_calls
            else:
                # Handle other potential types or None
                tool_calls_data = raw_tool_calls

            bg_color = role_colors.get(role, "#eeeeee")  # Default grey
            style = f"background-color: {bg_color}; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ddd;"

            html_parts.append(f'<div style="{style}">')
            # Display role without dashes
            html_parts.append(f"<b>{role.capitalize()}:</b><br>")

            if content:
                content_html = html.escape(content).replace("\\n", "<br>")
                html_parts.append(
                    f'<div style="white-space: pre-wrap; word-wrap: break-word; margin-top: 5px;">{content_html}</div>'
                )

            # Improved Tool Call Display
            if tool_calls_data:
                html_parts.append(
                    '<div style="margin-top: 8px; padding-top: 5px; border-top: 1px dashed #ccc;">'
                )
                html_parts.append("<b>Tool Calls:</b>")
                if isinstance(tool_calls_data, list) and all(
                    isinstance(tc, dict) for tc in tool_calls_data
                ):
                    html_parts.append(
                        '<ul style="margin: 5px 0 0 20px; padding-left: 0;">'
                    )
                    for tool_call in tool_calls_data:
                        func_info = tool_call.get(
                            "function", {}
                        )  # Get the inner 'function' dict
                        func_name = func_info.get("name", "Unknown Function")
                        # Get arguments, default to empty string if not found
                        args = func_info.get("arguments", "")
                        args_escaped = html.escape(args)
                        # Display function name and arguments
                        html_parts.append(
                            f"<li><code>{html.escape(func_name)}({args_escaped})</code></li>"
                        )
                    html_parts.append("</ul>")
                elif isinstance(
                    tool_calls_data, str
                ):  # Display raw string if parsing failed or it was just a string
                    html_parts.append(
                        f'<pre style="font-family: monospace; background-color: #eee; padding: 5px; margin-top: 5px; white-space: pre-wrap; word-wrap: break-word;">{html.escape(tool_calls_data)}</pre>'
                    )
                else:
                    # Fallback for unexpected format
                    html_parts.append(
                        f'<pre style="font-family: monospace; background-color: #eee; padding: 5px; margin-top: 5px; white-space: pre-wrap; word-wrap: break-word;">{html.escape(str(tool_calls_data))}</pre>'
                    )
                html_parts.append("</div>")

            html_parts.append("</div>")  # end message div

        html_parts.append("</div>")  # end chat history div
    else:
        html_parts.append("<p><i>No messages recorded for this interaction.</i></p>")

    html_parts.append("</div>")  # end main div

    display(HTML("".join(html_parts)))
