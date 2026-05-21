import html

from IPython.display import HTML, display


def display_image_grid(image_paths: list[str], images_per_row: int = 2):
    html = f"""
    <div style='display: grid; grid-template-columns: repeat({images_per_row}, 1fr); gap: 10px;'>
    """
    for path in image_paths:
        escaped_path = html.escape(path)
        html += f"<img src='{escaped_path}' style='max-width: 100%; height: auto;'>"
    html += "</div>"
    display(HTML(html))
