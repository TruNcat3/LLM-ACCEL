#!/usr/bin/env bash

# Shared profile resolver for the full-layer quantized controller, compute
# core, and C reference builds.  This file is sourceable; it deliberately
# does not change the caller's working directory or source a toolchain.

QUANTIZED_LAYER_PROFILE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
QUANTIZED_LAYER_CASE_ROOT="$(cd "${QUANTIZED_LAYER_PROFILE_SCRIPT_DIR}/.." && pwd -P)"

quantized_layer_profile_case_root() {
    printf '%s\n' "${QUANTIZED_LAYER_CASE_ROOT}"
}

quantized_layer_profile_file_sha256() {
    local path="$1"
    if [[ -s "${path}" ]]; then
        sha256sum "${path}" | awk '{print $1}'
    else
        printf 'MISSING\n'
    fi
}

# Return the source paths which can affect an HLS or Host artifact.  Keep the
# list explicit at directory boundaries so generated .build files can never
# become part of an artifact identity.  The full tree is intentional: HLS Tcl
# helpers and headers are transitive inputs even when a top-level translation
# unit does not include them directly.
quantized_layer_profile_source_paths() {
    local precision="${1:-${QUANTIZED_ALIGNMENT_PRECISION:-w4}}"
    local role="${2:-xo}" root="${QUANTIZED_LAYER_CASE_ROOT}"
    local host_source
    [[ "${precision}" == w4 || "${precision}" == w8 ]] || return 2
    [[ "${role}" == xo || "${role}" == host ]] || return 2
    (cd "${root}" && find kernel include tcl -type f \
        \( -name '*.cpp' -o -name '*.hpp' -o -name '*.tcl' \) -print)
    printf '%s\n' \
        'Makefile' \
        'scripts/quantized_layer_profiles.sh' \
        'scripts/run_vitis_hls.sh' \
        "scripts/build_vitis_quantized_${precision}_layer.sh"
    if [[ "${role}" == host ]]; then
        host_source="host/host_control_cache_quantized_${precision}_layer_hwemu.cpp"
        (cd "${root}" && find common/include -type f \
            \( -name '*.cpp' -o -name '*.hpp' \) -print)
        printf '%s\n' "${host_source}"
    fi
}

quantized_layer_profile_source_file_count() {
    quantized_layer_profile_source_paths "$@" | sort -u | wc -l | awk '{print $1}'
}

quantized_layer_profile_source_identity() {
    local precision="${1:-${QUANTIZED_ALIGNMENT_PRECISION:-w4}}"
    local role="${2:-xo}" root="${QUANTIZED_LAYER_CASE_ROOT}"
    (
        set -o pipefail
        cd "${root}" || exit
        mapfile -t source_paths < <(quantized_layer_profile_source_paths "${precision}" "${role}" | LC_ALL=C sort -u)
        [[ ${#source_paths[@]} -gt 0 ]] || exit 2
        sha256sum -- "${source_paths[@]}" | sha256sum | awk '{print $1}'
    )
}

quantized_layer_profile_source_file_records() {
    local precision="${1:-${QUANTIZED_ALIGNMENT_PRECISION:-w4}}"
    local role="${2:-xo}" root="${QUANTIZED_LAYER_CASE_ROOT}" rel
    while IFS= read -r rel; do
        [[ -n "${rel}" ]] || continue
        printf 'source_file=%s\t%s\n' "${rel}" \
            "$(quantized_layer_profile_file_sha256 "${root}/${rel}")"
    done < <(quantized_layer_profile_source_paths "${precision}" "${role}" | sort -u)
}

quantized_layer_profile_profile_helper_sha256() {
    local shell_helper="${QUANTIZED_LAYER_PROFILE_SCRIPT_DIR}/quantized_layer_profiles.sh"
    local tcl_helper="${QUANTIZED_LAYER_CASE_ROOT}/tcl/quantized_layer_profile.tcl"
    [[ -r "${shell_helper}" && -r "${tcl_helper}" ]] || return 2
    {
        printf 'shell=%s\n' "$(sha256sum "${shell_helper}" | awk '{print $1}')"
        printf 'tcl=%s\n' "$(sha256sum "${tcl_helper}" | awk '{print $1}')"
    } | sha256sum | awk '{print $1}'
}

quantized_layer_profile_build_inputs_sha256() {
    local precision="${1:-${QUANTIZED_ALIGNMENT_PRECISION:-w4}}"
    local kind="${2:-xo}" name path
    [[ "${precision}" == w4 || "${precision}" == w8 ]] || return 2
    [[ "${kind}" == xo || "${kind}" == xclbin || "${kind}" == host ]] || return 2
    {
        printf 'kind=%s\n' "${kind}"
        printf 'precision=%s\n' "${precision}"
        printf 'profile=%s\n' "${QUANTIZED_LAYER_PROFILE:-}"
        printf 'attention_variant=%s\n' "${QUANTIZED_LAYER_ATTENTION_VARIANT:-}"
        printf 'profile_cflags_sha256=%s\n' "${QUANTIZED_LAYER_PROFILE_CFLAGS_SHA256:-}"
        # Effective values are stable across a direct shell and tmux's explicit
        # environment forwarding. THREADS is a job limit, not a design input.
        for name in QUANTIZED_LAYER_ARTIFACT_TARGET QUANTIZED_LAYER_ARTIFACT_DEVICE \
            QUANTIZED_LAYER_ARTIFACT_FREQUENCY; do
            printf '%s=%s\n' "${name}" "${!name:-}"
        done
        if [[ "${kind}" == xclbin ]]; then
            for path in "${QUANTIZED_LAYER_ARTIFACT_PLATFORM:-}" \
                        "${QUANTIZED_LAYER_ARTIFACT_CONN_CFG:-}"; do
                if [[ -n "${path}" ]]; then
                    printf 'input=%s\t%s\n' "${path}" \
                        "$(quantized_layer_profile_file_sha256 "${path}")"
                else
                    printf 'input=\tMISSING\n'
                fi
            done
        fi
    } | sha256sum | awk '{print $1}'
}

quantized_layer_profile_die() {
    echo "quantized layer profile: $*" >&2
    return 2
}

quantized_layer_profile_source_sha256() {
    quantized_layer_profile_profile_helper_sha256
}

quantized_layer_profile_is_managed() {
    local name="$1" managed=" ${QUANTIZED_LAYER_PROFILE_MANAGED_NAMES:-} "
    [[ "${managed}" == *" ${name} "* ]]
}

quantized_layer_profile_check_input() {
    local name="$1" expected="$2" actual
    # Values exported by a previous invocation of this helper are not a new
    # user override. This also lets a test process resolve more than one
    # profile in sequence without leaking a false conflict.
    quantized_layer_profile_is_managed "${name}" && return 0
    [[ ${!name+x} == x ]] || return 0
    actual="${!name}"
    [[ -z "${actual}" || "${actual}" == "${expected}" ]] || {
        quantized_layer_profile_die "${name}=${actual} conflicts with profile value ${expected}"
        return 2
    }
}

quantized_layer_profile_apply() {
    local precision="${1:-${QUANTIZED_ALIGNMENT_PRECISION:-w4}}"
    local requested_profile="${QUANTIZED_LAYER_PROFILE:-}"
    local requested_variant="${QUANTIZED_LAYER_ATTENTION_VARIANT:-}"
    local requested_silu_lanes="${QUANTIZED_LAYER_SILU_LANES:-}"
    local requested_rms_lanes="${QUANTIZED_LAYER_RMS_LANES:-}"
    local requested_weight_outstanding="${QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING:-}"
    local requested_attention_wave="${QUANTIZED_LAYER_ATTENTION_WAVE:-}"
    local requested_prefill_ffn_overlap="${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP:-}"
    local profile variant silu_lanes silu_lanes_requested=0
    local rms_lanes rms_lanes_requested=0
    local weight_outstanding attention_wave prefill_ffn_overlap
    local weight_outstanding_requested=0 attention_wave_requested=0
    local prefill_ffn_overlap_requested=0
    local name value expected
    local -a names

    [[ "${precision}" == w4 || "${precision}" == w8 ]] || {
        quantized_layer_profile_die "precision must be w4 or w8"
        return 2
    }
    case "${requested_variant}" in
        ""|baseline|combined4) ;;
        *) quantized_layer_profile_die \
            "QUANTIZED_LAYER_ATTENTION_VARIANT must be baseline or combined4"; return 2 ;;
    esac
    if [[ -z "${requested_profile}" ]]; then
        case "${requested_variant}" in
            combined4) profile=attention ;;
            ""|baseline) profile=legacy ;;
        esac
    else
        profile="${requested_profile}"
    fi
    case "${profile}" in
        legacy|attention|attention_wave|pipeline|integrated) ;;
        *) quantized_layer_profile_die \
            "QUANTIZED_LAYER_PROFILE must be legacy, attention, attention_wave, pipeline, or integrated"; return 2 ;;
    esac
    variant=baseline
    [[ "${profile}" == legacy ]] || variant=combined4
    if [[ -n "${requested_variant}" && "${requested_variant}" != "${variant}" ]]; then
        quantized_layer_profile_die \
            "QUANTIZED_LAYER_PROFILE=${profile} conflicts with QUANTIZED_LAYER_ATTENTION_VARIANT=${requested_variant}"
        return 2
    fi
    if [[ "${precision}" == w4 && ${QUANTIZED_ALIGNMENT_W8+x} == x &&
          -n "${QUANTIZED_ALIGNMENT_W8}" ]] &&
       ! quantized_layer_profile_is_managed QUANTIZED_ALIGNMENT_W8; then
        quantized_layer_profile_die \
            "QUANTIZED_ALIGNMENT_W8=${QUANTIZED_ALIGNMENT_W8} conflicts with precision ${precision}"
        return 2
    fi

    case "${requested_silu_lanes}" in
        "") silu_lanes=1 ;;
        1|2|4)
            silu_lanes="${requested_silu_lanes}"
            # The resolver exports the canonical value below. Preserve the
            # explicitness bit across repeated calls so a generated default
            # does not become a user override when a caller resets profiles.
            silu_lanes_requested="${QUANTIZED_LAYER_SILU_LANES_REQUESTED:-1}"
            ;;
        *) quantized_layer_profile_die \
            "QUANTIZED_LAYER_SILU_LANES must be 1, 2, or 4"; return 2 ;;
    esac
    case "${requested_rms_lanes}" in
        "") rms_lanes=1 ;;
        1|2|4)
            rms_lanes="${requested_rms_lanes}"
            # Preserve the explicitness bit across repeated calls, just as
            # the SiLU control does for generated child-process defaults.
            rms_lanes_requested="${QUANTIZED_LAYER_RMS_LANES_REQUESTED:-1}"
            ;;
        *) quantized_layer_profile_die \
            "QUANTIZED_LAYER_RMS_LANES must be 1, 2, or 4"; return 2 ;;
    esac

    # The resolver exports canonical control values so child processes can
    # reconstruct the same profile.  The requested bits distinguish those
    # generated defaults from an explicit caller override on repeat calls.
    if [[ "${QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING_REQUESTED:-1}" == 0 ]]; then
        requested_weight_outstanding=""
    elif [[ -n "${requested_weight_outstanding}" ]]; then
        weight_outstanding_requested=1
    fi
    case "${requested_weight_outstanding}" in
        "") weight_outstanding=16 ;;
        16|32) weight_outstanding="${requested_weight_outstanding}" ;;
        *) quantized_layer_profile_die \
            "QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING must be 16 or 32"; return 2 ;;
    esac
    if [[ "${QUANTIZED_LAYER_ATTENTION_WAVE_REQUESTED:-1}" == 0 ]]; then
        requested_attention_wave=""
    elif [[ -n "${requested_attention_wave}" ]]; then
        attention_wave_requested=1
    fi
    case "${requested_attention_wave}" in
        "") attention_wave=0 ;;
        0|1) attention_wave="${requested_attention_wave}" ;;
        *) quantized_layer_profile_die \
            "QUANTIZED_LAYER_ATTENTION_WAVE must be 0 or 1"; return 2 ;;
    esac
    if [[ "${attention_wave_requested}" == 1 && "${profile}" == legacy ]]; then
        quantized_layer_profile_die \
            "QUANTIZED_LAYER_ATTENTION_WAVE override is only valid with attention, attention_wave, pipeline, or integrated"; return 2
    fi
    if [[ "${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED:-1}" == 0 ]]; then
        requested_prefill_ffn_overlap=""
    elif [[ -n "${requested_prefill_ffn_overlap}" ]]; then
        prefill_ffn_overlap_requested=1
    fi
    case "${requested_prefill_ffn_overlap}" in
        "") prefill_ffn_overlap=0 ;;
        0|1) prefill_ffn_overlap="${requested_prefill_ffn_overlap}" ;;
        *) quantized_layer_profile_die \
            "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP must be 0 or 1"; return 2 ;;
    esac
    if [[ "${prefill_ffn_overlap_requested}" == 1 &&
          "${prefill_ffn_overlap}" == 1 &&
          "${profile}" != pipeline && "${profile}" != integrated ]]; then
        quantized_layer_profile_die \
            "QUANTIZED_LAYER_PREFILL_FFN_OVERLAP=1 is only valid with pipeline or integrated profile"; return 2
    fi

    # Every value is emitted for every profile. This prevents a stale macro
    # inherited from an experiment from silently selecting a hybrid build.
    declare -A values=()
    values[QUANTIZED_ATTENTION_VALID_TAIL]=0
    values[QUANTIZED_ATTENTION_TILE_PREFETCH]=0
    values[QUANTIZED_ATTENTION_PROBABILITY_FUSE]=0
    values[QUANTIZED_ATTENTION_PACK_LANES]=1
    values[QUANTIZED_ATTENTION_ACTIVE_STATE]=0
    values[QUANTIZED_ATTENTION_HEAD_LANES]=1
    values[QUANTIZED_ATTENTION_WAVE_PIPELINE]=0
    values[QUANTIZED_BLOCK_PIPELINE]=0
    values[QUANTIZED_ASYNC_COMPUTE]=0
    values[QUANTIZED_LAYER_INTEGRATED_DECODE]=0
    values[QUANTIZED_DECODE_ROWS_MERGED]=0
    values[QDR_COMPACT]=0
    values[QDR_INGRESS_BITS]=4096
    values[QDR_SHARED_STATE]=0
    values[QDR_DIRECT_WIDE]=0
    values[QUANTIZED_DECODE_FFN_OVERLAP]=0
    values[QUANTIZED_PREFILL_FFN_OVERLAP]=0
    values[QUANTIZED_DECODE_SCALE_PREFETCH]=0
    values[QDR_BLOCK_OPS]=0
    values[QUANTIZED_DECODE_ROWS_ADAPTER_ACTIVATION_FIFO_DEPTH]=2
    values[QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING]=16
    values[QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST]=16
    values[CU_NL_LANES_CONFIG]="${silu_lanes}"
    values[CU_RMS_LANES_CONFIG]="${rms_lanes}"
    [[ "${precision}" == w8 ]] && values[QUANTIZED_ALIGNMENT_W8]=1

    if [[ "${profile}" != legacy ]]; then
        values[QUANTIZED_ATTENTION_VALID_TAIL]=1
        values[QUANTIZED_ATTENTION_TILE_PREFETCH]=1
        values[QUANTIZED_ATTENTION_PROBABILITY_FUSE]=1
        values[QUANTIZED_ATTENTION_PACK_LANES]=4
        values[QUANTIZED_ATTENTION_ACTIVE_STATE]=1
        values[QUANTIZED_ATTENTION_HEAD_LANES]=4
        values[QUANTIZED_ATTENTION_WAVE_PIPELINE]=0
    fi
    [[ "${profile}" == attention_wave ]] && values[QUANTIZED_ATTENTION_WAVE_PIPELINE]=1
    if [[ "${profile}" == pipeline || "${profile}" == integrated ]]; then
        values[QUANTIZED_BLOCK_PIPELINE]=1
        values[QUANTIZED_ASYNC_COMPUTE]=1
    fi
    if [[ "${profile}" == integrated ]]; then
        values[AP_INT_MAX_W]=4096
        values[QUANTIZED_LAYER_INTEGRATED_DECODE]=1
        values[QUANTIZED_DECODE_ROWS_MERGED]=1
        values[QDR_COMPACT]=1
        values[QDR_INGRESS_BITS]=4096
        values[QDR_SHARED_STATE]=1
        values[QDR_DIRECT_WIDE]=1
        values[QUANTIZED_DECODE_FFN_OVERLAP]=1
        values[QUANTIZED_DECODE_SCALE_PREFETCH]=1
        values[QDR_BLOCK_OPS]=1
        [[ "${precision}" == w8 ]] && values[QDR_BLOCK_OPS]=0
        values[QUANTIZED_DECODE_ROWS_ADAPTER_ACTIVATION_FIFO_DEPTH]=8
        values[QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING]=32
    fi
    [[ "${weight_outstanding_requested}" == 1 ]] && \
        values[QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING]="${weight_outstanding}"
    [[ "${attention_wave_requested}" == 1 ]] && \
        values[QUANTIZED_ATTENTION_WAVE_PIPELINE]="${attention_wave}"
    [[ "${prefill_ffn_overlap_requested}" == 1 ]] && \
        values[QUANTIZED_PREFILL_FFN_OVERLAP]="${prefill_ffn_overlap}"

    names=(
        QUANTIZED_ATTENTION_VALID_TAIL
        QUANTIZED_ATTENTION_TILE_PREFETCH
        QUANTIZED_ATTENTION_PROBABILITY_FUSE
        QUANTIZED_ATTENTION_PACK_LANES
        QUANTIZED_ATTENTION_ACTIVE_STATE
        QUANTIZED_ATTENTION_HEAD_LANES
        QUANTIZED_ATTENTION_WAVE_PIPELINE
        QUANTIZED_BLOCK_PIPELINE
        QUANTIZED_ASYNC_COMPUTE
        QUANTIZED_LAYER_INTEGRATED_DECODE
        QUANTIZED_DECODE_ROWS_MERGED
        QDR_COMPACT
        QDR_INGRESS_BITS
        QDR_SHARED_STATE
        QDR_DIRECT_WIDE
        QUANTIZED_DECODE_FFN_OVERLAP
        QUANTIZED_PREFILL_FFN_OVERLAP
        QUANTIZED_DECODE_SCALE_PREFETCH
        QDR_BLOCK_OPS
        QUANTIZED_DECODE_ROWS_ADAPTER_ACTIVATION_FIFO_DEPTH
        QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING
        QUANTIZED_BATCH_WEIGHT_MAX_READ_BURST
        CU_NL_LANES_CONFIG
        CU_RMS_LANES_CONFIG
    )
    [[ "${profile}" == integrated ]] && names+=(AP_INT_MAX_W)
    [[ "${precision}" == w8 ]] && names+=(QUANTIZED_ALIGNMENT_W8)
    for name in "${names[@]}"; do
        value="${values[${name}]}"
        quantized_layer_profile_check_input "${name}" "${value}" || return
    done

    export QUANTIZED_LAYER_PROFILE="${profile}"
    if [[ -n "${requested_profile}" ]]; then
        export QUANTIZED_LAYER_PROFILE_REQUESTED=1
    else
        export QUANTIZED_LAYER_PROFILE_REQUESTED=0
    fi
    export QUANTIZED_LAYER_ATTENTION_VARIANT="${variant}"
    export QUANTIZED_ALIGNMENT_PRECISION="${precision}"
    export QUANTIZED_LAYER_SILU_LANES="${silu_lanes}"
    export QUANTIZED_LAYER_SILU_LANES_REQUESTED="${silu_lanes_requested}"
    export QUANTIZED_LAYER_RMS_LANES="${rms_lanes}"
    export QUANTIZED_LAYER_RMS_LANES_REQUESTED="${rms_lanes_requested}"
    export QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING="${values[QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING]}"
    export QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING_REQUESTED="${weight_outstanding_requested}"
    export QUANTIZED_LAYER_ATTENTION_WAVE="${values[QUANTIZED_ATTENTION_WAVE_PIPELINE]}"
    export QUANTIZED_LAYER_ATTENTION_WAVE_REQUESTED="${attention_wave_requested}"
    export QUANTIZED_LAYER_PREFILL_FFN_OVERLAP="${values[QUANTIZED_PREFILL_FFN_OVERLAP]}"
    export QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED="${prefill_ffn_overlap_requested}"
    unset QUANTIZED_ALIGNMENT_W8
    QUANTIZED_LAYER_PROFILE_CFLAGS=()
    QUANTIZED_LAYER_PROFILE_MANAGED_NAMES=" QUANTIZED_LAYER_ATTENTION_VARIANT"
    for name in "${names[@]}"; do
        value="${values[${name}]}"
        export "${name}=${value}"
        QUANTIZED_LAYER_PROFILE_CFLAGS+=("-D${name}=${value}")
        QUANTIZED_LAYER_PROFILE_MANAGED_NAMES+=" ${name}"
    done
    export QUANTIZED_LAYER_PROFILE_MANAGED_NAMES
    export QUANTIZED_LAYER_PROFILE_SOURCE_SHA256="$(quantized_layer_profile_source_sha256)"
    export QUANTIZED_LAYER_SOURCE_IDENTITY_SHA256="$(
        quantized_layer_profile_source_identity "${precision}" xo
    )"
    export QUANTIZED_LAYER_PROFILE_CFLAGS_SHA256="$(printf '%s\n' "${QUANTIZED_LAYER_PROFILE_CFLAGS[@]}" | sha256sum | awk '{print $1}')"
}

quantized_layer_profile_cflags() {
    quantized_layer_profile_apply "${1:-${QUANTIZED_ALIGNMENT_PRECISION:-w4}}" || return
    printf '%s\n' "${QUANTIZED_LAYER_PROFILE_CFLAGS[@]}"
}

quantized_layer_profile_manifest_for_xo() {
    [[ $# == 1 ]] || { quantized_layer_profile_die "manifest helper takes one XO path"; return 2; }
    printf '%s.profile.manifest\n' "$1"
}

# Artifact identities bind the current source closure, selected build inputs,
# and generated bytes. Older profile-only manifests require an explicit rebuild.
quantized_layer_profile_manifest_for_xclbin() {
    [[ $# == 1 ]] || { quantized_layer_profile_die "manifest helper takes one xclbin path"; return 2; }
    printf '%s.link.manifest\n' "$1"
}

quantized_layer_profile_manifest_for_host() {
    [[ $# == 1 ]] || { quantized_layer_profile_die "manifest helper takes one Host path"; return 2; }
    printf '%s.host.manifest\n' "$1"
}

quantized_layer_profile_manifest_value() {
    local manifest="$1" key="$2"
    sed -n "s/^${key}=//p" "${manifest}" | head -n 1
}

quantized_layer_profile_link_inputs_sha256() {
    local controller_xo="$1" compute_xo="$2" platform="$3" conn_cfg="$4"
    {
        printf 'controller_xo=%s\t%s\n' "${controller_xo}" \
            "$(quantized_layer_profile_file_sha256 "${controller_xo}")"
        printf 'compute_xo=%s\t%s\n' "${compute_xo}" \
            "$(quantized_layer_profile_file_sha256 "${compute_xo}")"
        printf 'platform=%s\t%s\n' "${platform}" \
            "$(quantized_layer_profile_file_sha256 "${platform}")"
        printf 'conn_cfg=%s\t%s\n' "${conn_cfg}" \
            "$(quantized_layer_profile_file_sha256 "${conn_cfg}")"
    } | sha256sum | awk '{print $1}'
}

quantized_layer_profile_check_xo_reuse() {
    local precision="$1" xo="$2" manifest schema manifest_precision
    local profile source_digest flags_digest source_identity build_inputs xo_digest
    local expected_silu expected_rms expected_outstanding expected_wave expected_overlap
    [[ "${precision}" == w4 || "${precision}" == w8 ]] || return 2
    [[ -s "${xo}" ]] || return 0
    manifest="$(quantized_layer_profile_manifest_for_xo "${xo}")"
    [[ -s "${manifest}" ]] || {
        quantized_layer_profile_die \
            "XO identity manifest missing or empty: ${manifest}; rebuild explicitly with controller-xo or compute-xo"
        return 2
    }
    schema="$(quantized_layer_profile_manifest_value "${manifest}" schema)"
    [[ "${schema}" == quantized_layer_profile_xo_v2 ]] || {
        quantized_layer_profile_die \
            "unsupported/incomplete XO identity manifest: ${manifest}; rebuild with controller-xo or compute-xo"
        return 2
    }
    manifest_precision="$(quantized_layer_profile_manifest_value "${manifest}" precision)"
    [[ "${manifest_precision}" == "${precision}" ]] || {
        quantized_layer_profile_die \
            "XO precision mismatch for ${xo}: manifest=${manifest_precision:-missing} requested=${precision}; rebuild it"
        return 2
    }
    profile="$(quantized_layer_profile_manifest_value "${manifest}" profile)"
    [[ "${profile}" == "${QUANTIZED_LAYER_PROFILE}" ]] || {
        quantized_layer_profile_die \
            "XO profile mismatch for ${xo}: manifest=${profile:-missing} requested=${QUANTIZED_LAYER_PROFILE}; rebuild it"
        return 2
    }
    expected_silu="${CU_NL_LANES_CONFIG:-1}"
    expected_rms="${CU_RMS_LANES_CONFIG:-1}"
    expected_outstanding="${QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING:-16}"
    expected_wave="${QUANTIZED_ATTENTION_WAVE_PIPELINE:-0}"
    expected_overlap="${QUANTIZED_PREFILL_FFN_OVERLAP:-0}"
    for pair in \
        "silu_lanes=${expected_silu}" \
        "rms_lanes=${expected_rms}" \
        "weight_read_outstanding=${expected_outstanding}" \
        "attention_wave=${expected_wave}" \
        "prefill_ffn_overlap=${expected_overlap}"; do
        local key="${pair%%=*}" expected="${pair#*=}" actual
        actual="$(quantized_layer_profile_manifest_value "${manifest}" "${key}")"
        [[ "${actual}" == "${expected}" ]] || {
            quantized_layer_profile_die \
                "XO ${key} mismatch for ${xo}: manifest=${actual:-missing} requested=${expected}; rebuild it"
            return 2
        }
    done
    source_digest="$(quantized_layer_profile_manifest_value "${manifest}" profile_source_sha256)"
    [[ "${source_digest}" == "${QUANTIZED_LAYER_PROFILE_SOURCE_SHA256}" ]] || {
        quantized_layer_profile_die "profile helper source changed for ${xo}; rebuild it"
        return 2
    }
    flags_digest="$(quantized_layer_profile_manifest_value "${manifest}" profile_cflags_sha256)"
    [[ "${flags_digest}" == "${QUANTIZED_LAYER_PROFILE_CFLAGS_SHA256}" ]] || {
        quantized_layer_profile_die "profile flags changed for ${xo}; rebuild it"
        return 2
    }
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" source_file_count)" =~ ^[1-9][0-9]*$ ]] || {
        quantized_layer_profile_die \
            "XO source closure is missing from ${manifest}; rebuild explicitly with controller-xo or compute-xo"
        return 2
    }
    source_identity="$(quantized_layer_profile_source_identity "${precision}" xo)"
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" source_identity_sha256)" == "${source_identity}" ]] || {
        quantized_layer_profile_die \
            "XO source inputs changed for ${xo}; rebuild it with controller-xo or compute-xo"
        return 2
    }
    build_inputs="$(quantized_layer_profile_build_inputs_sha256 "${precision}" xo)"
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" build_inputs_sha256)" == "${build_inputs}" ]] || {
        quantized_layer_profile_die \
            "XO build/profile inputs changed for ${xo}; rebuild it with controller-xo or compute-xo"
        return 2
    }
    xo_digest="$(quantized_layer_profile_manifest_value "${manifest}" xo_sha256)"
    [[ "${xo_digest}" == "$(sha256sum "${xo}" | awk '{print $1}')" ]] || {
        quantized_layer_profile_die "XO bytes do not match its identity manifest: ${xo}; rebuild it"
        return 2
    }
}

quantized_layer_profile_write_xo_manifest() {
    local precision="$1" xo="$2" manifest source_identity build_inputs
    manifest="$(quantized_layer_profile_manifest_for_xo "${xo}")"
    [[ -s "${xo}" ]] || {
        quantized_layer_profile_die "cannot manifest missing XO: ${xo}"
        return 2
    }
    source_identity="$(quantized_layer_profile_source_identity "${precision}" xo)"
    build_inputs="$(quantized_layer_profile_build_inputs_sha256 "${precision}" xo)"
    {
        printf 'schema=quantized_layer_profile_xo_v2\n'
        printf 'precision=%s\n' "${precision}"
        printf 'profile=%s\n' "${QUANTIZED_LAYER_PROFILE}"
        printf 'attention_variant=%s\n' "${QUANTIZED_LAYER_ATTENTION_VARIANT}"
        printf 'silu_lanes=%s\n' "${CU_NL_LANES_CONFIG}"
        printf 'rms_lanes=%s\n' "${CU_RMS_LANES_CONFIG}"
        printf 'weight_read_outstanding=%s\n' "${QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING}"
        printf 'attention_wave=%s\n' "${QUANTIZED_ATTENTION_WAVE_PIPELINE}"
        printf 'prefill_ffn_overlap=%s\n' "${QUANTIZED_PREFILL_FFN_OVERLAP}"
        printf 'profile_source_sha256=%s\n' "${QUANTIZED_LAYER_PROFILE_SOURCE_SHA256}"
        printf 'profile_cflags_sha256=%s\n' "${QUANTIZED_LAYER_PROFILE_CFLAGS_SHA256}"
        printf 'source_identity_sha256=%s\n' "${source_identity}"
        printf 'source_file_count=%s\n' "$(quantized_layer_profile_source_file_count "${precision}" xo)"
        printf 'build_inputs_sha256=%s\n' "${build_inputs}"
        printf 'xo_sha256=%s\n' "$(sha256sum "${xo}" | awk '{print $1}')"
    } >"${manifest}"
}

quantized_layer_profile_check_xclbin_reuse() {
    local precision="$1" xclbin="$2" controller_xo="$3" compute_xo="$4"
    local platform="$5" conn_cfg="$6" manifest schema manifest_precision
    local source_identity build_inputs link_inputs
    [[ "${precision}" == w4 || "${precision}" == w8 ]] || return 2
    [[ -s "${xclbin}" ]] || {
        quantized_layer_profile_die "missing xclbin: ${xclbin}; link it before run"
        return 2
    }
    manifest="$(quantized_layer_profile_manifest_for_xclbin "${xclbin}")"
    [[ -s "${manifest}" ]] || {
        quantized_layer_profile_die \
            "xclbin link manifest missing or empty: ${manifest}; relink before run"
        return 2
    }
    schema="$(quantized_layer_profile_manifest_value "${manifest}" schema)"
    [[ "${schema}" == quantized_layer_link_v1 ]] || {
        quantized_layer_profile_die "unsupported xclbin link manifest: ${manifest}; relink before run"
        return 2
    }
    manifest_precision="$(quantized_layer_profile_manifest_value "${manifest}" precision)"
    [[ "${manifest_precision}" == "${precision}" ]] || {
        quantized_layer_profile_die \
            "xclbin precision mismatch: manifest=${manifest_precision:-missing} requested=${precision}; relink it"
        return 2
    }
    quantized_layer_profile_check_xo_reuse "${precision}" "${controller_xo}" || return
    quantized_layer_profile_check_xo_reuse "${precision}" "${compute_xo}" || return
    for path in "${platform}" "${conn_cfg}"; do
        [[ -s "${path}" ]] || {
            quantized_layer_profile_die "missing current link input: ${path}; relink xclbin"
            return 2
        }
    done
    source_identity="$(quantized_layer_profile_source_identity "${precision}" xo)"
    build_inputs="$(quantized_layer_profile_build_inputs_sha256 "${precision}" xclbin)"
    link_inputs="$(quantized_layer_profile_link_inputs_sha256 \
        "${controller_xo}" "${compute_xo}" "${platform}" "${conn_cfg}")"
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" source_identity_sha256)" == "${source_identity}" ]] || {
        quantized_layer_profile_die "xclbin source inputs changed; relink ${xclbin}"
        return 2
    }
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" build_inputs_sha256)" == "${build_inputs}" ]] || {
        quantized_layer_profile_die "xclbin build/platform inputs changed; relink ${xclbin}"
        return 2
    }
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" link_inputs_sha256)" == "${link_inputs}" ]] || {
        quantized_layer_profile_die "XO/platform/connection inputs changed for ${xclbin}; relink it"
        return 2
    }
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" controller_xo_sha256)" == "$(quantized_layer_profile_file_sha256 "${controller_xo}")" ]] || {
        quantized_layer_profile_die "controller XO changed after link; relink ${xclbin}"
        return 2
    }
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" compute_xo_sha256)" == "$(quantized_layer_profile_file_sha256 "${compute_xo}")" ]] || {
        quantized_layer_profile_die "compute XO changed after link; relink ${xclbin}"
        return 2
    }
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" platform_sha256)" == "$(quantized_layer_profile_file_sha256 "${platform}")" ]] || {
        quantized_layer_profile_die "platform changed after link; relink ${xclbin}"
        return 2
    }
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" conn_cfg_sha256)" == "$(quantized_layer_profile_file_sha256 "${conn_cfg}")" ]] || {
        quantized_layer_profile_die "connection config changed after link; relink ${xclbin}"
        return 2
    }
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" xclbin_sha256)" == "$(sha256sum "${xclbin}" | awk '{print $1}')" ]] || {
        quantized_layer_profile_die "xclbin bytes do not match its link manifest; relink it"
        return 2
    }
}

quantized_layer_profile_write_xclbin_manifest() {
    local precision="$1" xclbin="$2" controller_xo="$3" compute_xo="$4"
    local platform="$5" conn_cfg="$6" manifest source_identity build_inputs link_inputs
    manifest="$(quantized_layer_profile_manifest_for_xclbin "${xclbin}")"
    for path in "${xclbin}" "${controller_xo}" "${compute_xo}" "${platform}" "${conn_cfg}"; do
        [[ -s "${path}" ]] || {
            quantized_layer_profile_die "cannot write link manifest; missing input: ${path}"
            return 2
        }
    done
    source_identity="$(quantized_layer_profile_source_identity "${precision}" xo)"
    build_inputs="$(quantized_layer_profile_build_inputs_sha256 "${precision}" xclbin)"
    link_inputs="$(quantized_layer_profile_link_inputs_sha256 \
        "${controller_xo}" "${compute_xo}" "${platform}" "${conn_cfg}")"
    {
        printf 'schema=quantized_layer_link_v1\n'
        printf 'precision=%s\n' "${precision}"
        printf 'profile=%s\n' "${QUANTIZED_LAYER_PROFILE}"
        printf 'attention_variant=%s\n' "${QUANTIZED_LAYER_ATTENTION_VARIANT}"
        printf 'source_identity_sha256=%s\n' "${source_identity}"
        printf 'build_inputs_sha256=%s\n' "${build_inputs}"
        printf 'link_inputs_sha256=%s\n' "${link_inputs}"
        printf 'controller_xo=%s\n' "${controller_xo}"
        printf 'controller_xo_sha256=%s\n' "$(quantized_layer_profile_file_sha256 "${controller_xo}")"
        printf 'compute_xo=%s\n' "${compute_xo}"
        printf 'compute_xo_sha256=%s\n' "$(quantized_layer_profile_file_sha256 "${compute_xo}")"
        printf 'platform=%s\n' "${platform}"
        printf 'platform_sha256=%s\n' "$(quantized_layer_profile_file_sha256 "${platform}")"
        printf 'conn_cfg=%s\n' "${conn_cfg}"
        printf 'conn_cfg_sha256=%s\n' "$(quantized_layer_profile_file_sha256 "${conn_cfg}")"
        printf 'xclbin_sha256=%s\n' "$(sha256sum "${xclbin}" | awk '{print $1}')"
    } >"${manifest}"
}

quantized_layer_profile_check_host_reuse() {
    local precision="$1" host_exe="$2" manifest schema source_identity build_inputs
    [[ -s "${host_exe}" ]] || {
        quantized_layer_profile_die "missing Host executable: ${host_exe}; build host before run"
        return 2
    }
    manifest="$(quantized_layer_profile_manifest_for_host "${host_exe}")"
    [[ -s "${manifest}" ]] || {
        quantized_layer_profile_die \
            "Host identity manifest missing or empty: ${manifest}; rebuild host before run"
        return 2
    }
    schema="$(quantized_layer_profile_manifest_value "${manifest}" schema)"
    [[ "${schema}" == quantized_layer_host_v1 ]] || {
        quantized_layer_profile_die "unsupported Host identity manifest: ${manifest}; rebuild host"
        return 2
    }
    source_identity="$(quantized_layer_profile_source_identity "${precision}" host)"
    build_inputs="$(quantized_layer_profile_build_inputs_sha256 "${precision}" host)"
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" source_identity_sha256)" == "${source_identity}" ]] || {
        quantized_layer_profile_die "Host source inputs changed; rebuild ${host_exe}"
        return 2
    }
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" build_inputs_sha256)" == "${build_inputs}" ]] || {
        quantized_layer_profile_die "Host build/profile inputs changed; rebuild ${host_exe}"
        return 2
    }
    [[ "$(quantized_layer_profile_manifest_value "${manifest}" host_sha256)" == "$(sha256sum "${host_exe}" | awk '{print $1}')" ]] || {
        quantized_layer_profile_die "Host bytes do not match its identity manifest; rebuild ${host_exe}"
        return 2
    }
}

quantized_layer_profile_write_host_manifest() {
    local precision="$1" host_exe="$2" manifest source_identity build_inputs
    manifest="$(quantized_layer_profile_manifest_for_host "${host_exe}")"
    [[ -s "${host_exe}" ]] || {
        quantized_layer_profile_die "cannot manifest missing Host executable: ${host_exe}"
        return 2
    }
    source_identity="$(quantized_layer_profile_source_identity "${precision}" host)"
    build_inputs="$(quantized_layer_profile_build_inputs_sha256 "${precision}" host)"
    {
        printf 'schema=quantized_layer_host_v1\n'
        printf 'precision=%s\n' "${precision}"
        printf 'profile=%s\n' "${QUANTIZED_LAYER_PROFILE}"
        printf 'attention_variant=%s\n' "${QUANTIZED_LAYER_ATTENTION_VARIANT}"
        printf 'source_identity_sha256=%s\n' "${source_identity}"
        printf 'build_inputs_sha256=%s\n' "${build_inputs}"
        printf 'host_sha256=%s\n' "$(sha256sum "${host_exe}" | awk '{print $1}')"
    } >"${manifest}"
}
