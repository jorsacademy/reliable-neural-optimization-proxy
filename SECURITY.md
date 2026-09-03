# Security policy

## Supported version

Security and correctness fixes are applied to the current `main` branch. Research snapshots should pin a commit SHA.

## Threat model

This package processes numerical JSON instances and NumPy dataset/checkpoint archives. It does not execute generated code, shell commands, model-supplied Python, or network requests.

Defensive controls include:

- strict dimensions and finite-number validation;
- `allow_pickle=False` for NumPy archive loading;
- versioned feature and checkpoint schemas;
- rejection of incompatible unit counts and tensor shapes;
- exact post-repair feasibility audits;
- certificate-oracle assertions in benchmarks;
- exact fallback when a certificate is invalid or too loose.

## Untrusted artifacts

Do not treat `.npz` files as authenticated merely because pickle loading is disabled. Large or intentionally malformed arrays may exhaust memory or CPU. In production, enforce file-size limits, isolate processing, authenticate artifacts, and verify cryptographic hashes.

A checkpoint can produce arbitrarily poor predictions. The reliability layer is designed to reject or repair poor numerical outputs, but it is not a defense against denial-of-service through oversized input, process compromise, dependency compromise, or malicious modification of the trusted repair/certificate code.

## Sensitive data

The repository's examples are synthetic. Do not commit proprietary costs, capacities, dispatch records, credentials, or personal data. Git history retains deleted secrets.

## Reporting

Report suspected vulnerabilities privately to the repository owner. Include the affected commit, reproduction steps, impact, and any relevant artifact hashes. Do not include confidential production data in a public issue.

## Deployment boundary

This is research software, not a safety-critical dispatch controller. Production use requires independent model validation, numerical monitoring, resource limits, access controls, dependency review, rollback procedures, and domain-specific operational safeguards.
