import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightLlmsTxt from "starlight-llms-txt";

const page = (slug, label) => ({ slug, label });

export default defineConfig({
  site: "https://art.openpipe.ai",
  integrations: [
    starlight({
      title: "ART",
      description: "Train LLMs to be better agents using reinforcement learning.",
      favicon: "/images/site-assets/favicon.png",
      logo: {
        light: "./public/images/site-assets/logo/light.svg",
        dark: "./public/images/site-assets/logo/dark.svg",
        alt: "OpenPipe",
        replacesTitle: true,
      },
      customCss: ["@fontsource-variable/inter", "./src/styles/custom.css"],
      expressiveCode: {
        // Match Mintlify's Shiki themes; the frame is restyled in custom.css.
        themes: ["github-light-default", "dark-plus"],
      },
      components: {
        Header: "./src/components/Header.astro",
        PageFrame: "./src/components/PageFrame.astro",
        PageTitle: "./src/components/PageTitle.astro",
        Sidebar: "./src/components/Sidebar.astro",
        TwoColumnContent: "./src/components/TwoColumnContent.astro",
      },
      head: [
        { tag: "script", attrs: { src: "/analytics.js", async: true } },
      ],
      social: [
        { icon: "discord", label: "Discord", href: "https://discord.gg/zbBHRUpwf4" },
        { icon: "github", label: "GitHub", href: "https://github.com/openpipe/ART" },
        { icon: "x.com", label: "X", href: "https://twitter.com/OpenPipeAI" },
        { icon: "linkedin", label: "LinkedIn", href: "https://www.linkedin.com/company/openpipe/about/" },
      ],
      editLink: {
        baseUrl: "https://github.com/OpenPipe/ART/edit/main/docs/src/content/docs/",
      },
      plugins: [starlightLlmsTxt()],
      sidebar: [
        {
          label: "Get Started",
          items: [
            page("getting-started/about", "ART Docs"),
            page("getting-started/quick-start", "Quick Start"),
            page("getting-started/installation-setup", "Installation + Setup"),
            page("getting-started/multi-node", "Multi-node deployment"),
            page("getting-started/notebooks", "Notebooks"),
            page("getting-started/faq", "FAQ"),
          ],
        },
        {
          label: "Fundamentals",
          items: [
            page("fundamentals/training-loop", "Training Loop"),
            page("fundamentals/art-client", "ART Client"),
            page("fundamentals/art-backend", "ART Backend"),
            page("fundamentals/ruler", "RULER"),
            page("fundamentals/sft-training", "SFT Training"),
          ],
        },
        {
          label: "Features",
          items: [
            page("features/checkpoint-forking", "Checkpoint Forking"),
            page("features/checkpoint-deletion", "Deleting Checkpoints"),
            page("features/additional-histories", "Additional Histories"),
            page("features/tracking-metrics", "Tracking Metrics"),
            page("features/mcp-rl", "MCP Training"),
          ],
        },
        {
          label: "Integrations",
          items: [
            page("integrations/langgraph-integration", "🦜🔗 LangGraph"),
            page("integrations/openenv-integration", "🌍 OpenEnv"),
          ],
        },
        {
          label: "Tutorials",
          items: [
            page("tutorials/summarizer", "Summarizer"),
            page("tutorials/open-deep-research", "Open Deep Research"),
          ],
        },
        {
          label: "Resources",
          items: [
            page("resources/models", "Models"),
            page("resources/glossary", "Glossary"),
          ],
        },
        {
          label: "Experimental",
          items: [page("experimental/gspo", "GSPO")],
        },
      ],
    }),
  ],
  redirects: {
    "/": "/getting-started/about",
  },
});
