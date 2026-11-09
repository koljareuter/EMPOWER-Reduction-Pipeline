#!/bin/bash
# Build KMOS DRP 4.5.11 with the KMOS_LCORR_DEGREE patch next to an existing KMOS 4.5.11 installation.
# The patch only lets kmos_sci_red read the polynomial degree of its OH-line wavelength correction from the
# environment variable KMOS_LCORR_DEGREE; without the variable the patched recipes give bit-identical results
# to the standard ones. The existing installation is only read.
#
#   PIPE=/path/to/kmos/install TARBALL=/path/to/kmos-4.5.11.tar.gz DEST=/path/to/KMOS_4.5.11_lcorr \
#       bash drp_patch/build_kmos_lcorr_drp.sh
#
#   PIPE     prefix of the existing KMOS 4.5.11 kit installation (contains bin/esorex, lib/libcplcore*)
#   TARBALL  kmos-4.5.11.tar.gz from that pipeline kit (md5 f8d1e09b7516834bb7edbbc69e3ec9e7)
#   DEST     new folder for the patched build
#
# Afterwards set recipe_dir in the [cubes] section of the config to the printed recipe directory.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
: "${PIPE:?set PIPE to the existing KMOS installation prefix}"
: "${TARBALL:?set TARBALL to kmos-4.5.11.tar.gz from the pipeline kit}"
: "${DEST:?set DEST to a new folder for the patched build}"
PY=${PY:-python3}
RECIPES=$DEST/install/lib/esopipes-plugins

[ -e "$DEST/install" ] && { echo "$DEST/install exists already (remove it or choose another DEST)"; exit 1; }
md5sum "$TARBALL"
mkdir -p "$DEST/src" && tar xzf "$TARBALL" -C "$DEST/src" || exit 1
$PY "$HERE/kmos_lcorr_degree_patch.py" "$DEST/src/kmos-4.5.11" || exit 1
cd "$DEST/src/kmos-4.5.11" || exit 1
{ ./configure --prefix="$DEST/install" --with-cpl="$PIPE" --with-telluriccorr="$PIPE" &&
  make -j4 && make install; } > "$DEST/build.log" 2>&1
echo "build exit $? (log $DEST/build.log)"
ls "$RECIPES/kmos-4.5.11/kmos_sci_red.so" || exit 1
lib=$(find "$DEST/install" -name 'libkmos.so.0' -o -name 'libkmos.0.dylib' | head -1)
strings "$lib" | grep -q KMOS_LCORR_DEGREE && echo "patch present in $lib" || { echo "PATCH MISSING in $lib"; exit 1; }
echo "recipe_dir for the config: $RECIPES"
