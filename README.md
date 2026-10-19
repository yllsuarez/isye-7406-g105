# isye-7406-g105

Monorepo for the ISYE 7406 group project, organized by solution (similar to a
Visual Studio solution with multiple projects):

| Folder | Purpose | Language |
|---|---|---|
| [`r-models/`](./r-models) | Train the ML model(s) | R |
| [`data-cleaning/`](./data-cleaning) | Data ingestion & preprocessing | Python |
| [`web-service/`](./web-service) | Serve model predictions via a simple web app | Python |

## Getting started

Open `isye-7406-g105.code-workspace` in VS Code to load all three folders as a
multi-root workspace, each with its own interpreter/settings.

See each folder's `README.md` for setup and run instructions.

## Project board

Progress is tracked on the [project board](https://github.com/users/yllsuarez/projects/2),
with a lane per solution (R Models, Data Cleaning, Web Service).
