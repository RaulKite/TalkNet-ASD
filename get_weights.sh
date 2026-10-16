#!/usr/bin/env bash
# Fetch and verify the two model weights this repository needs but does not carry.
#
#   ./get_weights.sh            download whatever is missing or wrong
#   ./get_weights.sh --check    only verify what is on disk; never touches the network
#   ./get_weights.sh --force    re-download even if the files already look right
#
# Why a script and not a clone: .gitignore keeps *.model and *.pth out of the
# repository on purpose. Committing them would add ~146 MB to every clone forever,
# and they never change with the code. Release assets keep them out of history while
# still giving a URL that anyone can fetch and checksum. See WEIGHTS.md.
set -euo pipefail

AQUI="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd -P)"

# Where the assets live. Point this at your own mirror if you host one; the
# checksums below still have to match, so a mirror cannot silently change the model.
BASE_URL="${ASD_WEIGHTS_URL:-https://github.com/raul-agent/TalkNet-ASD/releases/download/weights-e802c54}"

# ruta_relativa  sha256                                                             bytes
PESOS=(
  "pretrain_TalkSet.model d985ddd07dca08864a28337726f5b7d91b6426bebc7327f0e9b21cf9ff1cc937 63227827"
  "model/faceDetector/s3fd/sfd_face.pth d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491 89844381"
)

MODO=""
case "${1:-}" in
  "")          ;;
  --check)     MODO="comprobar" ;;
  --force)     MODO="forzar" ;;
  -h|--help)   sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *)           echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
esac

coincide() {  # coincide <ruta> <sha256 esperado>
  [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

faltan=0
for entrada in "${PESOS[@]}"; do
  read -r ruta sha tam <<<"$entrada"
  destino="$AQUI/$ruta"

  if coincide "$destino" "$sha"; then
    echo "  ok        $ruta"
    continue
  fi

  if [ "$MODO" = "comprobar" ]; then
    if [ -f "$destino" ]; then
      echo "  DIFFERS   $ruta  (expected sha256 $sha)" >&2
    else
      echo "  MISSING   $ruta  (expected $tam bytes, sha256 $sha)" >&2
    fi
    faltan=1
    continue
  fi

  command -v curl >/dev/null || { echo "curl is required to download weights" >&2; exit 1; }
  nombre="$(basename "$ruta")"
  tmp="$(mktemp)"
  echo "  fetching  $nombre"
  # A failed or partial download must not look like a weight: -f makes HTTP errors
  # non-zero and the checksum below rejects anything that is not the expected file.
  # In a terminal a progress bar is worth 86 MB of waiting; in a log or a pipe the
  # carriage returns become noise, so there it stays quiet and still reports errors.
  if [ -t 1 ]; then CURL_FLAGS=(-fL --progress-bar); else CURL_FLAGS=(-fL -sS); fi
  if ! curl "${CURL_FLAGS[@]}" -o "$tmp" "$BASE_URL/$nombre"; then
    echo "  could not download $nombre from $BASE_URL/$nombre" >&2
    echo "  if you host these files elsewhere: ASD_WEIGHTS_URL=<base> ./get_weights.sh" >&2
    rm -f "$tmp"; exit 1
  fi
  coincide "$tmp" "$sha" || {
    echo "  downloaded $nombre does not match the expected sha256, refusing to install it" >&2
    echo "  got $(sha256sum "$tmp" | cut -d' ' -f1) expected $sha" >&2
    rm -f "$tmp"; exit 1
  }
  mkdir -p "$(dirname "$destino")"
  mv "$tmp" "$destino"
  # mktemp creates mode 600. Keep the weight readable: a 600 checkpoint in a shared
  # checkout shows up much later as a failure inside the longest stage of a pipeline,
  # and copying one around with `cp -a` spreads the mode.
  chmod 0644 "$destino"
  echo "  installed $ruta  sha256 $(sha256sum "$destino" | cut -c1-16)..."
done

if [ "$faltan" -ne 0 ]; then
  echo
  echo "weights missing or wrong; run ./get_weights.sh to fetch them" >&2
  exit 1
fi
