# Security Policy

## Supported code

Security fixes are applied to the current default branch. Historical notebooks are retained for educational context and may depend on older third-party packages; they should not be used as production deployment templates without review.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's security-advisory feature when available. Do not open a public issue containing credentials, exploitable payloads, private data or detailed instructions that could put users at risk.

A useful report includes:

- affected file and version or commit;
- impact and realistic attack scenario;
- minimal reproduction steps;
- environment and dependency versions;
- suggested mitigation, when known.

## Secrets and data

Never commit API keys, access tokens, private datasets, personal information or proprietary model artefacts. Use environment variables or a dedicated secret manager for credentials.

## Model and dataset risks

Downloaded models and datasets are third-party content. Review their source, licence, access requirements and trust level before use. Avoid loading untrusted serialised Python objects. Prefer safe, documented model formats and verified upstream repositories.

## Production use

Examples in this repository are reference implementations. Before production deployment, add authentication, authorisation, input-size limits, rate limiting, dependency scanning, logging controls, privacy review and adversarial testing appropriate to the application.