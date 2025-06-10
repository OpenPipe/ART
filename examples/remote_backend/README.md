# Remote Backend Example

Start the ART server on a remote machine and specify where model weights and artifacts should be stored:

```bash
art run --path /mnt/art_storage
```

Then point your client at the server by setting `OPENPIPE_API_BASE` to its URL.
