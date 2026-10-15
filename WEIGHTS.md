# Model weights

Two weight files are required to run anything, and neither is in the repository:
`.gitignore` excludes `*.model` and `*.pth`.

| file | used for | bytes | sha256 |
|---|---|---|---|
| `pretrain_TalkSet.model` | the pretrained TalkNet detector (repo root) | 63227827 | `d985ddd07dca08864a28337726f5b7d91b6426bebc7327f0e9b21cf9ff1cc937` |
| `model/faceDetector/s3fd/sfd_face.pth` | the S3FD face detector | 89844381 | `d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491` |

## Get them

```bash
./get_weights.sh            # download whatever is missing or wrong
./get_weights.sh --check    # verify only; never touches the network
```

The script downloads both files, refuses to install anything whose SHA-256 differs from
the table above, and leaves them mode `644`.

## Why they are release assets

Committing 146 MB would put them in every clone and every `git fetch` forever, for files
that never change with the code — and git history cannot forget, so undoing it later means
rewriting it and breaking every fork. Release assets sit outside history and keep the
checksum visible. The pair is published at
[`raul-agent/TalkNet-ASD`, release `weights-e802c54`](https://github.com/raul-agent/TalkNet-ASD/releases/tag/weights-e802c54),
anchored to commit `e802c54d33bdf4fbdb6d3bcb24ffec9883d8ff02`. To use a different mirror,
set `ASD_WEIGHTS_URL`; the checksums still apply, so a mirror cannot quietly change a model.

Both files were distributed as part of this project (`LICENSE.md`, MIT, Copyright (c) 2021
Tao Ruijie), which permits redistribution.

## If you copy them from another machine

Use `cp -r`, not `cp -a`. Weight files are often mode `600` where they were first
downloaded, and `cp -a` preserves that mode — the result is a checkout that "has" the
weights and then fails to read them, typically in the middle of the longest stage of a
run. `./get_weights.sh --check` catches it in a second.
