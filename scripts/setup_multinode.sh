#!/bin/bash
set -euo pipefail

readonly ucx_version=1.21.0
readonly ucx_sha256=2374d2fcf3186fbfd5e27633ab153aabaeb6b4f503a88563d2aca67cf51ed2c1
readonly nixl_version=1.3.2
readonly nixl_commit=de8115ca97d3f8fb63a4988e9b4d4a038b2e0f72
readonly nixl_sha256=a9d88772935e91181733f00df0a7e93b6be5f1d29300e5c06cfc6b4ad2f6dbdb
readonly asio_sha256=12e7bb4dada8bc1191de9d550a59ee658ce4e645ffc97c911c099ab4e8699d55
readonly asio_patch_sha256=8bed3693016874b097e4d902c4ca8daf1b6abf1b5a56b0c5c02827d4e0747ddb
readonly etcd_version=3.5.33
readonly etcd_sha256=5025b5b24d81a9616b6e284ccd439b9a3df055ef8fdcdc142af3ec8f6a3b3c95

cache_dir=${ART_SETUP_CACHE_DIR:-$HOME/.cache/art/multinode}
build_root=$(mktemp -d /tmp/art-multinode-setup-XXXXXX)
test -d "$build_root"
trap 'rm -rf "$build_root"' EXIT
mkdir -p "$cache_dir"

download() {
    local url=$1 destination=$2 sha256=$3 lock_fd partial
    exec {lock_fd}>"$destination.lock"
    flock "$lock_fd"
    if ! printf '%s  %s\n' "$sha256" "$destination" | sha256sum -c - >/dev/null 2>&1; then
        partial=$(mktemp "$destination.partial.XXXXXX")
        if ! curl -fL --retry 3 -o "$partial" "$url" ||
            ! printf '%s  %s\n' "$sha256" "$partial" | sha256sum -c -; then
            rm -f "$partial"
            return 1
        fi
        mv "$partial" "$destination"
    fi
    exec {lock_fd}>&-
}

publish_dir() {
    local source=$1 destination=$2 marker=$3
    test -d "$source"
    if test -e "$destination"; then
        test -f "$destination/.art-version" && grep -Fxq "$marker" "$destination/.art-version" && return
        echo "Refusing to replace unowned or mismatched $destination" >&2
        exit 1
    fi
    sudo install -d "$destination"
    sudo cp -a "$source/." "$destination/"
    printf '%s\n' "$marker" | sudo tee "$destination/.art-version" >/dev/null
}

link_install() {
    local alias=$1 target=$2
    if test -e "$alias" && ! test -L "$alias"; then
        cmp -s "$alias" "$target" && return
        echo "Refusing to replace non-symlink $alias" >&2
        exit 1
    fi
    sudo ln -sfn "$target" "$alias"
}

assert_linked_from() {
    local object=$1 library=$2 prefix=$3 resolved
    resolved=$(ldd "$object" | awk -v library="$library" '$1 == library {print $3}')
    resolved=$(readlink -f "$resolved")
    case "$resolved" in
        "$prefix"/*) ;;
        *) echo "$object resolves $library outside $prefix: ${resolved:-missing}" >&2; exit 1 ;;
    esac
}

for command in gcc make cmake ninja patchelf pkg-config python3; do
    command -v "$command" >/dev/null || {
        echo "Cluster bootstrap must provide $command" >&2
        exit 1
    }
done
test "$(uname -m)" = x86_64
test -f /usr/include/infiniband/verbs.h
grep -q 'Open Kernel Module' /proc/driver/nvidia/version
grep -q '^nvidia_peermem ' /proc/modules
grep -q '^EnableStreamMemOPs: 1$' /proc/driver/nvidia/params
grep -q 'PeerMappingOverride=1' /proc/driver/nvidia/params
test -c /dev/infiniband/rdma_cm
test -c /dev/infiniband/uverbs0
compgen -G '/sys/class/infiniband/*' >/dev/null

cuda_version=$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([0-9]*\.[0-9]*\).*/\1/p' /usr/local/cuda/version.json | head -1)
cuda_major=${cuda_version%%.*}
case "$cuda_major" in
    12)
        nixl_wheel_url=https://files.pythonhosted.org/packages/f8/d3/2964339654b3fe85e7aa62fdce4da3b97ee40337f3b72466aa79251f1196/nixl_cu12-1.3.2-cp312-cp312-manylinux_2_28_x86_64.whl
        nixl_wheel_sha256=ef8ccdffcd54978e8a799de59287efcb3af0b7ba3bf02e04bc4df4c842f1f569
        ;;
    13)
        nixl_wheel_url=https://files.pythonhosted.org/packages/99/d8/5768b907b85d8856c07674ddd0ffeb736ed987ff530a12e8f17a273cab3b/nixl_cu13-1.3.2-cp312-cp312-manylinux_2_28_x86_64.whl
        nixl_wheel_sha256=22fcd7183b2cd831b3da781c9e9991c5f6ef77a238ee6b7ac05d42558ea469a9
        ;;
    *) echo "CUDA 12 or 13 is required, found ${cuda_version:-unknown}" >&2; exit 1 ;;
esac
gpu_arches=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | sed 's/\.//' | sort -u | paste -sd,)
test -n "$gpu_arches"
compiler_id=$(gcc -dumpfullversion | cut -d. -f1)
art_prefix=/usr/local/art-multinode
if test -e "$art_prefix"; then
    test -f "$art_prefix/.art-owner" && grep -Fxq art-multinode "$art_prefix/.art-owner" || {
        echo "Refusing to use unowned $art_prefix" >&2
        exit 1
    }
else
    sudo install -d "$art_prefix/bin"
    printf 'art-multinode\n' | sudo tee "$art_prefix/.art-owner" >/dev/null
fi

download \
    "https://github.com/etcd-io/etcd/releases/download/v${etcd_version}/etcd-v${etcd_version}-linux-amd64.tar.gz" \
    "$cache_dir/etcd-v${etcd_version}-linux-amd64.tar.gz" "$etcd_sha256"
tar -xzf "$cache_dir/etcd-v${etcd_version}-linux-amd64.tar.gz" -C "$build_root"
etcd_prefix="/usr/local/etcd-${etcd_version}"
etcd_marker="etcd=${etcd_version} sha256=${etcd_sha256}"
etcd_stage="$build_root/etcd-stage"
install -d "$etcd_stage/bin"
install -m 0755 "$build_root/etcd-v${etcd_version}-linux-amd64/etcd" "$etcd_stage/bin/etcd"
install -m 0755 "$build_root/etcd-v${etcd_version}-linux-amd64/etcdctl" "$etcd_stage/bin/etcdctl"
publish_dir "$etcd_stage" "$etcd_prefix" "$etcd_marker"
link_install "$art_prefix/bin/etcd" "$etcd_prefix/bin/etcd"
link_install "$art_prefix/bin/etcdctl" "$etcd_prefix/bin/etcdctl"

ucx_prefix="/usr/local/ucx-${ucx_version}-cuda${cuda_version}-${ucx_sha256:0:12}"
ucx_marker="ucx=${ucx_version} sha256=${ucx_sha256} cuda=${cuda_version}"
if ! test -f "$ucx_prefix/.art-version" || ! grep -Fxq "$ucx_marker" "$ucx_prefix/.art-version"; then
    test ! -e "$ucx_prefix" || {
        echo "Refusing to replace mismatched $ucx_prefix" >&2
        exit 1
    }
    download \
        "https://github.com/openucx/ucx/releases/download/v${ucx_version}/ucx-${ucx_version}.tar.gz" \
        "$cache_dir/ucx-${ucx_version}.tar.gz" "$ucx_sha256"
    tar -xzf "$cache_dir/ucx-${ucx_version}.tar.gz" -C "$build_root"
    (
        cd "$build_root/ucx-${ucx_version}"
        ./configure --prefix="$ucx_prefix" \
            --disable-logging --disable-debug --disable-assertions \
            --disable-params-check --enable-mt --enable-shared --disable-static \
            --disable-doxygen-doc --enable-optimizations --enable-cma \
            --enable-devel-headers --with-cuda=/usr/local/cuda --with-verbs \
            --with-dm --without-gdrcopy
        make -j"$(nproc)"
        make DESTDIR="$build_root/ucx-stage" install
    )
    publish_dir "$build_root/ucx-stage$ucx_prefix" "$ucx_prefix" "$ucx_marker"
fi
link_install "$art_prefix/ucx" "$ucx_prefix"

download "$nixl_wheel_url" "$cache_dir/nixl-cu${cuda_major}-${nixl_version}.whl" "$nixl_wheel_sha256"
download \
    "https://github.com/ai-dynamo/nixl/archive/${nixl_commit}.tar.gz" \
    "$cache_dir/nixl-${nixl_commit}.tar.gz" "$nixl_sha256"
nixl_source="$build_root/nixl-${nixl_commit}"
tar -xzf "$cache_dir/nixl-${nixl_commit}.tar.gz" -C "$build_root"
python3 -m zipfile -e "$cache_dir/nixl-cu${cuda_major}-${nixl_version}.whl" "$build_root/nixl-wheel"
wheel_core=$(find "$build_root/nixl-wheel" -maxdepth 1 -type d -name ".nixl_cu${cuda_major}.mesonpy.libs" -print -quit)
wheel_deps="$build_root/nixl-wheel/nixl_cu${cuda_major}.libs"
test -f "$wheel_core/libnixl.so"
test -d "$wheel_deps"

core_prefix="/usr/local/nixl-${nixl_version}-cu${cuda_major}-${nixl_wheel_sha256:0:12}-layout1"
core_marker="nixl=${nixl_version} commit=${nixl_commit} wheel_sha256=${nixl_wheel_sha256} cuda=${cuda_version} layout=1"
core_stage="$build_root/nixl-core-stage"
install -d "$core_stage/include/gpu/ucx" "$core_stage/lib/x86_64-linux-gnu" "$core_stage/lib/nixl_cu${cuda_major}.libs"
cp -a "$wheel_core/." "$core_stage/lib/x86_64-linux-gnu/"
cp -a "$wheel_deps/." "$core_stage/lib/nixl_cu${cuda_major}.libs/"
install -m 0644 "$nixl_source/src/api/cpp"/nixl{,_descriptors,_params,_types}.h "$core_stage/include/"
install -m 0644 "$nixl_source/src/api/gpu/ucx/nixl_device.cuh" "$core_stage/include/gpu/ucx/"
publish_dir "$core_stage" "$core_prefix" "$core_marker"
link_install "$art_prefix/nixl" "$core_prefix"

plugin_prefix="/usr/local/nixl-ucx-${nixl_version}-${nixl_commit:0:12}-ucx${ucx_version}-cuda${cuda_version}-gcc${compiler_id}-sm${gpu_arches//,/+}-abi1"
plugin_marker="nixl=${nixl_version} commit=${nixl_commit} ucx=${ucx_version} ucx_sha256=${ucx_sha256} cuda=${cuda_version} gcc=${compiler_id} sm=${gpu_arches} rpath=1"
if ! test -f "$plugin_prefix/.art-version" || ! grep -Fxq "$plugin_marker" "$plugin_prefix/.art-version"; then
    test ! -e "$plugin_prefix" || {
        echo "Refusing to replace mismatched $plugin_prefix" >&2
        exit 1
    }
    package_cache="$nixl_source/subprojects/packagecache"
    mkdir -p "$package_cache"
    download \
        https://github.com/mesonbuild/wrapdb/releases/download/asio_1.30.2-2/asio-1.30.2.tar.gz \
        "$package_cache/asio-1.30.2.tar.gz" "$asio_sha256"
    download \
        https://wrapdb.mesonbuild.com/v2/asio_1.30.2-2/get_patch \
        "$package_cache/asio_1.30.2-2_patch.zip" "$asio_patch_sha256"
    build_dir="$build_root/nixl-build"
    PKG_CONFIG_PATH="$ucx_prefix/lib/pkgconfig" \
        uvx --from meson==1.9.1 --with pybind11==2.13.6 meson setup "$build_dir" "$nixl_source" \
        --buildtype=release --prefix="$plugin_prefix" --libdir=lib \
        -Ducx_path="$ucx_prefix" -Denable_plugins=UCX \
        -Dbuild_tests=false -Dbuild_examples=false -Dinstall_headers=false \
        -Dnixl_cuda_arch_list="$gpu_arches"
    uvx --from meson==1.9.1 --with pybind11==2.13.6 meson compile -C "$build_dir" \
        UCX -j "$(nproc)"
    plugin_stage="$build_root/nixl-plugin-stage"
    install -d "$plugin_stage/lib/plugins" "$plugin_stage/lib"
    install -m 0755 "$build_dir/src/plugins/ucx/libplugin_UCX.so" "$plugin_stage/lib/plugins/"
    install -m 0755 "$build_dir/src/utils/common/libnixl_common.so" "$plugin_stage/lib/"
    install -m 0755 "$build_dir/src/infra/libnixl_build.so" "$plugin_stage/lib/"
    install -m 0755 "$build_dir/src/utils/serdes/libserdes.so" "$plugin_stage/lib/"
    patchelf --set-rpath "\$ORIGIN/..:$ucx_prefix/lib" "$plugin_stage/lib/plugins/libplugin_UCX.so"
    for library in "$plugin_stage/lib"/*.so; do
        patchelf --set-rpath '\$ORIGIN' "$library"
    done
    publish_dir "$plugin_stage" "$plugin_prefix" "$plugin_marker"
fi
link_install "$art_prefix/nixl-ucx" "$plugin_prefix"

plugin="$art_prefix/nixl-ucx/lib/plugins/libplugin_UCX.so"
for object in "$plugin" "$art_prefix"/nixl-ucx/lib/*.so; do
    ! ldd "$object" | grep -q 'not found'
done
! ldd "$core_prefix/lib/x86_64-linux-gnu/libnixl.so" | grep -q 'not found'
for library in libucp.so.0 libucs.so.0 libuct.so.0 libucm.so.0; do
    assert_linked_from "$plugin" "$library" "$ucx_prefix"
done
for library in libnixl_common.so libnixl_build.so libserdes.so; do
    assert_linked_from "$plugin" "$library" "$plugin_prefix"
done
rc_gda_count=$(
    UCX_IB_GDA_RETAIN_INACTIVE_CTX=yes UCX_MODULE_DIR="$ucx_prefix/lib/ucx" \
        "$ucx_prefix/bin/ucx_info" -d 2>/dev/null \
        | awk '/Transport: rc_gda/{count++} END{print count+0}'
)
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
test "$rc_gda_count" -ge "$gpu_count"
"$etcd_prefix/bin/etcd" --version | grep -q "etcd Version: ${etcd_version}"
printf 'ART multinode dependencies ready: CUDA %s, SM %s, %s rc_gda resources\n' \
    "$cuda_version" "$gpu_arches" "$rc_gda_count"
