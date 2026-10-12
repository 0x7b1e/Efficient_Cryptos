#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo $0" >&2
  exit 1
fi

# Install the reconnect script and the Anycast runtime as root-owned files, so
# a NOPASSWD sudo rule cannot be abused by editing user-writable files.
src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/reconnect_anycast.sh"
anycast_src="${ANYCAST_SRC:-/home/${SUDO_USER:-root}/Anycast}"
anycast_dest="/usr/local/lib/shadow-anycast"
dest="/usr/local/sbin/shadow-reconnect-anycast"
if [ ! -d "$anycast_src" ]; then
  echo "missing Anycast directory: $anycast_src" >&2
  exit 1
fi
install -d -o root -g root -m 0755 "$anycast_dest"
cp -a "$anycast_src"/. "$anycast_dest"/
chown -R root:root "$anycast_dest"
install -o root -g root -m 0755 "$src" "$dest"
echo "installed $dest as root:root 0755"
echo "installed Anycast runtime at $anycast_dest as root:root"
echo "the live runner uses 'sudo -n $dest' by default when --proxy is set"
echo "sudoers example: ${SUDO_USER:-user} ALL=(root) NOPASSWD: $dest"
