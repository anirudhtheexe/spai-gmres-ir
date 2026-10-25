# Archived experimental results

All original result files are preserved at their original relative paths in
`../archives/research_20261019.zip`. The archive also contains data, provenance,
historical campaign tooling and reports. Run from the project root:

```sh
python scripts/research_archive.py verify
python scripts/research_archive.py restore
```

The source submission builds and runs its small regression tests independently
of those historical experiments. This directory remains the historical root
experiment-lock location if research tooling is restored.
