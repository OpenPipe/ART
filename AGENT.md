## uv package manager by default

This project uses the `uv` package manager.

- To add a dependency, run `uv add <package>`.
- To run a script, run `uv run <script>`.
- To examine dependencies, consult the `pyproject.toml` file.

## Testing

- Always run tests before committing. The test command is `uv run prek run --all-files`.

## Releases

- If asked to help with a release, refer to the checklist in CONTRIBUTING.md. Be sure to first share a draft of the release notes with the user before actually publishing the release to GitHub.
- To trigger the release workflow via GitHub CLI: `gh workflow run create-draft-release.yml --field version_type=patch` (use `minor` or `major` instead of `patch` as needed)

## Documentation

- All documentation is in the `docs` directory. Pages are in `docs/src/content/docs`.
- Use Node.js 22.12+ and pnpm. Run `pnpm install --frozen-lockfile` and `pnpm build` from `docs/` to validate changes.
- If you add a new page, be sure to add it to the sidebar in `docs/src/navigation.ts`.
- If you move a page, be sure to update the sidebar in `docs/src/navigation.ts` and check for any broken links.

### Adding images

- Add images to the `docs/public/images` directory
- If the image is a png, first convert it to webp using `magick <input.png> <output.webp>`. Do not include the original png in the repo.
- Use the `<Frame>` tag to add images with captions as seen in the page `checkpoint-forking.mdx`.

### Adding notes

- Add notes using the `<Note>` tag as seen in the page `ruler.mdx`
