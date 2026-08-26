# ART Documentation

This directory contains the source for the ART documentation website hosted at [https://art.openpipe.ai](https://art.openpipe.ai). It is a static Astro site that can be deployed to any static host.

## Prerequisites

- [pnpm](https://pnpm.io/installation)
- [Node.js](https://nodejs.org/en/download/)

## Local development

1. Navigate to the `docs` directory.
2. Run `pnpm install`.
3. Run `pnpm dev` to start the development server on port 3001.
4. Edit pages in `src/content/docs`.

Edits are reflected immediately by the development server.

### Adding new pages

1. Create a new `.md` or `.mdx` file in `src/content/docs`.
2. Add the page to the appropriate group in `src/navigation.ts`.

### Building and deploying

Run `pnpm build` before opening a pull request. This checks the project, builds the static pages, and creates the Pagefind search index. The output in `dist` can be hosted on Cloudflare Workers, Cloudflare Pages, GitHub Pages, or another static host.

The included `wrangler.jsonc` targets the `art-docs` Cloudflare Pages project in the OpenPipe account. After authenticating Wrangler with that account, run `pnpm deploy`.
