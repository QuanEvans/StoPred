#!/usr/bin/env bash
#
# Download or update the wwPDB mmCIF archive in the native rsync layout.
#
# Usage:
#   bash external/download_pdb_mmcif.sh /path/to/download
#
# Output layout:
#   /path/to/download/pdb_mmcif/divided/
#   /path/to/download/pdb_mmcif/obsolete/
#
# Files remain compressed as .cif.gz. Use parse_PDBmmcif_gz.py for parsing.
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

if ! command -v rsync &> /dev/null ; then
    echo "Error: rsync could not be found. Please install rsync."
    exit 1
fi

DOWNLOAD_DIR="$1"
ROOT_DIR="${DOWNLOAD_DIR}/pdb_mmcif"
DIVIDED_DIR="${ROOT_DIR}/divided"
OBSOLETE_DIR="${ROOT_DIR}/obsolete"

PDB_PORT="${PDB_PORT:-33444}"
PDB_SERVER="${PDB_SERVER:-rsync.rcsb.org::ftp}"
PDB_RSYNC_ROOT="${PDB_RSYNC_ROOT:-${PDB_SERVER}/data/structures}"

echo "Updating PDB mmCIF archive in: ${ROOT_DIR}"
echo "Rsync root: ${PDB_RSYNC_ROOT}"
echo "Port: ${PDB_PORT}"
echo
echo "If the download speed is too slow, try one of the wwPDB mirrors:"
echo "  PDB_RSYNC_ROOT=rsync.ebi.ac.uk::pub/databases/pdb/data/structures"
echo "  PDB_RSYNC_ROOT=ftp.pdbj.org::ftp_data/structures"
echo "See https://www.wwpdb.org/ftp/pdb-ftp-sites for more options."
echo

mkdir -p "${DIVIDED_DIR}" "${OBSOLETE_DIR}"

echo "[1/2] Updating divided/mmCIF ..."
rsync -rlpt -v -z --delete --port="${PDB_PORT}" \
  "${PDB_RSYNC_ROOT}/divided/mmCIF/" \
  "${DIVIDED_DIR}/"

echo
echo "[2/2] Updating obsolete/mmCIF ..."
rsync -rlpt -v -z --delete --port="${PDB_PORT}" \
  "${PDB_RSYNC_ROOT}/obsolete/mmCIF/" \
  "${OBSOLETE_DIR}/"

echo
echo "Done. Parse with:"
echo "  python parse_PDBmmcif_gz.py ${ROOT_DIR} --exclude_obsolete -o raw_data/PDBmmcif.json"
