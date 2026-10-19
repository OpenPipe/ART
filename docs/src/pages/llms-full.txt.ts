import { getCollection } from "astro:content";
import { pages } from "../navigation";

export async function GET() {
  const entries = await getCollection("docs");
  const bySlug = new Map(entries.map((entry) => [entry.id.replace(/\.(md|mdx)$/, ""), entry]));
  const sections = pages.flatMap((page) => {
    const entry = bySlug.get(page.slug);
    return entry ? [`# ${entry.data.title}`, "", entry.body, ""] : [];
  });
  return new Response(["# ART Documentation", "", ...sections].join("\n"), { headers: { "Content-Type": "text/plain; charset=utf-8" } });
}
