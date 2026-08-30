# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/scattercode/tetrak-hy-trainer/security/advisories/new)
(Security tab → "Report a vulnerability"). Do not open a public issue for a
security problem.

We aim to acknowledge reports within seven days. This is a small
collaborative project, not a company with a security team — but we take
reports seriously and will keep you informed as we investigate.

## Supported versions

Only the latest release receives fixes. There are no maintenance branches.

## What we do ourselves

- The full resolved dependency tree is pinned in `uv.lock` and scanned by
  Trivy on every push and weekly on a schedule; Dependabot keeps the lock
  and the workflow actions current.
- Trained weights are published only as GitHub Release assets with SHA-256
  checksums and a provenance record. Verify the checksum before loading
  weights — a `.pth` file is a pickle, and loading an untrusted one
  executes code.
