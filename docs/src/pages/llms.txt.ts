import { pages, hrefFor } from "../navigation";

export function GET() {
  const lines = [
    "# ART Documentation",
    "",
    "> Train LLMs to be better agents using reinforcement learning.",
    "",
    ...pages.map((page) => `- [${page.label}](https://art.openpipe.ai${hrefFor(page.slug)})`),
  ];
  return new Response(lines.join("\n"), { headers: { "Content-Type": "text/plain; charset=utf-8" } });
}
