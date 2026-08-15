# Model files

Place trained detector weights in this directory. Model binaries (`*.pt`,
`*.onnx`, `*.engine`, and similar files) are intentionally excluded from Git.

The default local profile in `configs/models.yaml` expects:

```text
models/six_class_final.pt
```

Keep production weights in an approved artifact store or Git LFS and verify
their checksum before deployment.
