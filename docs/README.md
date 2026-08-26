# ART Documentation

This directory contains the source for the ART documentation website hosted at [https://art.openpipe.ai](https://art.openpipe.ai). The site is built with [Astro Starlight](https://starlight.astro.build/) and can be deployed to any static host.

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
2. Add the page to the appropriate sidebar group in `astro.config.mjs`.

### Building and deploying

Run `pnpm build` before opening a pull request. The static output in `dist` can be hosted on Cloudflare Workers, Cloudflare Pages, GitHub Pages, or another static host.

The included `wrangler.jsonc` is ready for Cloudflare Workers static-asset hosting. After authenticating Wrangler with the intended Cloudflare account, run `pnpm deploy`.
