export interface NavigationItem {
  label: string;
  slug: string;
  icon?: string;
}

export interface NavigationGroup {
  label: string;
  items: NavigationItem[];
}

export const navigation: NavigationGroup[] = [
  {
    label: "Get Started",
    items: [
      { label: "ART Docs", slug: "getting-started/about", icon: "house" },
      { label: "Quick Start", slug: "getting-started/quick-start", icon: "forward" },
      { label: "Installation + Setup", slug: "getting-started/installation-setup", icon: "gear" },
      { label: "Multi-node deployment", slug: "getting-started/multi-node", icon: "network-wired" },
      { label: "Notebooks", slug: "getting-started/notebooks", icon: "book" },
      { label: "FAQ", slug: "getting-started/faq", icon: "block-question" },
    ],
  },
  {
    label: "Fundamentals",
    items: [
      { label: "Training Loop", slug: "fundamentals/training-loop", icon: "recycle" },
      { label: "ART Client", slug: "fundamentals/art-client", icon: "laptop-code" },
      { label: "ART Backend", slug: "fundamentals/art-backend", icon: "server" },
      { label: "RULER", slug: "fundamentals/ruler", icon: "ruler" },
      { label: "SFT Training", slug: "fundamentals/sft-training", icon: "graduation-cap" },
    ],
  },
  {
    label: "Features",
    items: [
      { label: "Checkpoint Forking", slug: "features/checkpoint-forking" },
      { label: "Deleting Checkpoints", slug: "features/checkpoint-deletion" },
      { label: "Additional Histories", slug: "features/additional-histories" },
      { label: "Tracking Metrics", slug: "features/tracking-metrics", icon: "chart-line" },
      { label: "MCP Training", slug: "features/mcp-rl" },
    ],
  },
  {
    label: "Integrations",
    items: [
      { label: "🦜🔗 LangGraph", slug: "integrations/langgraph-integration" },
      { label: "🌍 OpenEnv", slug: "integrations/openenv-integration" },
    ],
  },
  {
    label: "Tutorials",
    items: [
      { label: "Summarizer", slug: "tutorials/summarizer", icon: "list" },
      { label: "Open Deep Research", slug: "tutorials/open-deep-research", icon: "magnifying-glass" },
    ],
  },
  {
    label: "Resources",
    items: [
      { label: "Models", slug: "resources/models", icon: "robot" },
      { label: "Glossary", slug: "resources/glossary", icon: "circle-info" },
    ],
  },
  {
    label: "Experimental",
    items: [{ label: "GSPO", slug: "experimental/gspo" }],
  },
];

export const pages = navigation.flatMap((group) =>
  group.items.map((item) => ({ ...item, section: group.label })),
);

export const hrefFor = (slug: string) => `/${slug}/`;

export function pageFor(slug: string) {
  return pages.find((page) => page.slug === slug);
}

export function neighborsFor(slug: string) {
  const index = pages.findIndex((page) => page.slug === slug);
  return {
    previous: index > 0 ? pages[index - 1] : undefined,
    next: index >= 0 && index < pages.length - 1 ? pages[index + 1] : undefined,
  };
}
