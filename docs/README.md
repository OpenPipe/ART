# OpenPipe Documentation

This repository contains the source for the ART documentation website hosted at [https://art.openpipe.ai](https://art.openpipe.ai).

## Prerequisites

Ensure you have the following packages installed on your machine:

- [pnpm](https://pnpm.io/installation)
- [node](https://nodejs.org/en/download/)

## Contributing

To edit the documentation follow these steps:

1. Clone the repository
2. Navigate to the `docs` directory
3. Run `pnpm install` to install the dependencies
4. Run `pnpm dev` to start the development server
5. Edit the files in the `docs` directory

Edits to files should immediately be reflected in the development server.

### Adding new pages

1. Create a new .mdx file in the `docs` directory
2. Navigate to the `mint.json` file and add the new page to the appropriate section to the `navigation` array, or create a new section. Ensure that the path to the new page is correct.

### Deploying changes

To deploy changes to the hosted docs, commit your changes in a new git branch and create a pull request. Once the pull request is merged, the changes will be deployed to the hosted docs.
