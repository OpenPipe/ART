import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import expressiveCode from "astro-expressive-code";

export default defineConfig({
  site: "https://art.openpipe.ai",
  integrations: [
    expressiveCode({
      themes: ["github-light-default", "dark-plus"],
    }),
    mdx(),
    sitemap(),
  ],
});
