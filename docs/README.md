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

### CoreWeave deployment

`wrangler.coreweave.jsonc` defines the `openpipe-art-docs` Worker with Static
Assets. It serves `dist` on the CoreWeave account's assigned `workers.dev`
hostname, including the custom 404 page. It does not bind `art.openpipe.ai` or
deploy to the OpenPipe `art-docs` Pages project. Canonicals, sitemap URLs, and
`llms.txt` links retain `art.openpipe.ai` until the domain cutover.

The GitHub **Docs** workflow builds docs on relevant pull requests and pushes
to `main`. A successful build on `main` automatically deploys the same artifact
to CoreWeave. **Run workflow** on `main` can also redeploy without a new commit.
PR builds have no Cloudflare credentials and cannot deploy.

Configure the GitHub environment **cloudflare-docs** before merging:

- Secret `CLOUDFLARE_API_TOKEN`: a token scoped to that account with
  **Account / Workers Scripts / Edit** permission. No DNS permission is needed.
- Restrict environment deployment branches to `main`. Leave required reviewers
  unset so documentation deployments do not require manual approval.

The workflow creates or updates the Worker. No custom domain or DNS change is
required to test it. If the account has not enabled a `workers.dev` subdomain,
an account administrator must configure one first. Validate the temporary URL,
including search and existing deep links, before moving `art.openpipe.ai`.

Use the workflow for deployments; no dashboard upload or local deployment is
needed. CI uses the locked Wrangler version and does not depend on a personal
Cloudflare login. Never commit `CLOUDFLARE_API_TOKEN`.
The deployment config pins the CoreWeave account ID
`81820de800df4910fe9d0efd69585ddf` so it cannot accidentally target OpenPipe.
