# isye-7406-g105

Monorepo for the ISYE 7406 group project, organized by solution (similar to a
Visual Studio solution with multiple projects):

| Folder | Purpose | Language |
|---|---|---|
| [`r-models/`](./r-models) | Train the ML model (LassoCV) | Python |
| [`data-cleaning/`](./data-cleaning) | Data ingestion & preprocessing | Python |
| [`web-service/`](./web-service) | FastAPI backend + React/TypeScript single-page app | Python / TypeScript |

## Getting started

Open `isye-7406-g105.code-workspace` in VS Code to load all three folders as a
multi-root workspace, each with its own interpreter/settings.

See each folder's `README.md` for setup and run instructions.

## Project board

Progress is tracked on the [project board](https://github.com/users/yllsuarez/projects/2),
with a lane per solution (Models, Data Cleaning, Web Service).
