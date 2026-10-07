#!/usr/bin/env bash

# Shared profile resolver for the full-layer quantized controller, compute
# core, and C reference builds.  This file is sourceable; it deliberately
# does not change the caller's working directory or source a toolchain.

quantized_layer_profile_die() {
    echo "quantized layer profile: $*" >&2
    return 2
}

quantized_layer_profile_source_sha256() {
    local shell_helper tcl_helper
    shell_helper="$(realpath "${BASH_SOURCE[0]}")"
    tcl_helper="$(dirname "${shell_helper}")/../tcl/quantized_layer_profile.tcl"
    [[ -r "${tcl_helper}" ]] || {
        quantized_layer_profile_die "missing Tcl profile helper: ${tcl_helper}"
        return 2
    }
    {
        printf 'shell=%s\n' "$(sha256sum "${shell_helper}" | awk '{print $1}')"
        printf 'tcl=%s\n' "$(sha256sum "${tcl_helper}" | awk '{print $1}')"
    } | sha256sum | awk '{print $1}'
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

quantized_layer_profile_check_xo_reuse() {
    local precision="$1" xo="$2" manifest schema manifest_precision profile
    local source_digest flags_digest xo_digest manifest_silu_lanes expected_silu_lanes
    local manifest_rms_lanes expected_rms_lanes default_rms_lanes
    local manifest_weight_outstanding manifest_attention_wave
    local manifest_prefill_ffn_overlap
    local expected_weight_outstanding expected_attention_wave
    local expected_prefill_ffn_overlap
    local default_weight_outstanding default_attention_wave
    local default_prefill_ffn_overlap
    [[ "${precision}" == w4 || "${precision}" == w8 ]] || return 2
    expected_silu_lanes="${CU_NL_LANES_CONFIG:-1}"
    expected_rms_lanes="${CU_RMS_LANES_CONFIG:-1}"
    expected_weight_outstanding="${QUANTIZED_BATCH_WEIGHT_READ_OUTSTANDING:-16}"
    expected_attention_wave="${QUANTIZED_ATTENTION_WAVE_PIPELINE:-0}"
    expected_prefill_ffn_overlap="${QUANTIZED_PREFILL_FFN_OVERLAP:-0}"
    default_weight_outstanding=16
    [[ "${QUANTIZED_LAYER_PROFILE}" == integrated ]] && default_weight_outstanding=32
    default_attention_wave=0
    [[ "${QUANTIZED_LAYER_PROFILE}" == attention_wave ]] && default_attention_wave=1
    default_prefill_ffn_overlap=0
    default_rms_lanes=1
    # A stale manifest is harmless when the artifact itself is absent: the
    # caller will synthesize a replacement before attempting to link it.
    [[ -s "${xo}" ]] || return 0
    manifest="$(quantized_layer_profile_manifest_for_xo "${xo}")"
    if [[ ! -e "${manifest}" ]]; then
        if [[ -s "${xo}" ]]; then
            echo "profile_identity=unverified_legacy_xo precision=${precision} xo=${xo}" >&2
            export QUANTIZED_LAYER_PROFILE_UNVERIFIED_XO=1
            if [[ "${QUANTIZED_LAYER_PROFILE_REQUESTED:-0}" == 1 &&
                  "${QUANTIZED_LAYER_PROFILE}" == attention_wave ]]; then
                quantized_layer_profile_die \
                    "cannot reuse XO without a profile manifest for explicit profile ${QUANTIZED_LAYER_PROFILE}: ${xo}"
                return 2
            fi
            if [[ "${QUANTIZED_LAYER_PROFILE_REQUESTED:-0}" == 1 &&
                  ( "${QUANTIZED_LAYER_PROFILE}" == pipeline ||
                    "${QUANTIZED_LAYER_PROFILE}" == integrated ) &&
                  "${QUANTIZED_LAYER_ALLOW_LEGACY_XO_REUSE:-0}" != 1 ]]; then
                quantized_layer_profile_die \
                    "cannot reuse XO without a profile manifest for explicit profile ${QUANTIZED_LAYER_PROFILE}: ${xo}"
                return 2
            fi
            if [[ "${QUANTIZED_LAYER_SILU_LANES_REQUESTED:-0}" == 1 ||
                  "${expected_silu_lanes}" != 1 ]]; then
                quantized_layer_profile_die \
                    "cannot reuse XO without a profile manifest for explicit SiLU lanes=${expected_silu_lanes}: ${xo}"
                return 2
            fi
            if [[ "${QUANTIZED_LAYER_RMS_LANES_REQUESTED:-0}" == 1 ||
                  "${expected_rms_lanes}" != "${default_rms_lanes}" ]]; then
                quantized_layer_profile_die \
                    "cannot reuse XO without a profile manifest for explicit RMS lanes=${expected_rms_lanes}: ${xo}"
                return 2
            fi
            if [[ "${QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING_REQUESTED:-0}" == 1 ||
                  "${expected_weight_outstanding}" != "${default_weight_outstanding}" ]]; then
                quantized_layer_profile_die \
                    "cannot reuse XO without a profile manifest for explicit weight read outstanding=${expected_weight_outstanding}: ${xo}"
                return 2
            fi
            if [[ "${QUANTIZED_LAYER_ATTENTION_WAVE_REQUESTED:-0}" == 1 ||
                  "${expected_attention_wave}" != "${default_attention_wave}" ]]; then
                quantized_layer_profile_die \
                    "cannot reuse XO without a profile manifest for explicit attention wave=${expected_attention_wave}: ${xo}"
                return 2
            fi
            if [[ "${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED:-0}" == 1 ||
                  "${expected_prefill_ffn_overlap}" != "${default_prefill_ffn_overlap}" ]]; then
                quantized_layer_profile_die \
                    "cannot reuse XO without a profile manifest for explicit Prefill FFN overlap=${expected_prefill_ffn_overlap}: ${xo}"
                return 2
            fi
        fi
        return 0
    fi
    [[ -s "${manifest}" ]] || {
        quantized_layer_profile_die "empty XO profile manifest: ${manifest}"
        return 2
    }
    schema="$(sed -n 's/^schema=//p' "${manifest}" | head -n 1)"
    [[ "${schema}" == quantized_layer_profile_v1 ]] || {
        quantized_layer_profile_die "unsupported XO profile manifest: ${manifest}"
        return 2
    }
    manifest_precision="$(sed -n 's/^precision=//p' "${manifest}" | head -n 1)"
    [[ "${manifest_precision}" == "${precision}" ]] || {
        quantized_layer_profile_die \
            "XO precision mismatch for ${xo}: manifest=${manifest_precision:-missing} requested=${precision}"
        return 2
    }
    profile="$(sed -n 's/^profile=//p' "${manifest}" | head -n 1)"
    manifest_silu_lanes="$(sed -n 's/^silu_lanes=//p' "${manifest}" | head -n 1)"
    manifest_rms_lanes="$(sed -n 's/^rms_lanes=//p' "${manifest}" | head -n 1)"
    manifest_weight_outstanding="$(sed -n 's/^weight_read_outstanding=//p' "${manifest}" | head -n 1)"
    manifest_attention_wave="$(sed -n 's/^attention_wave=//p' "${manifest}" | head -n 1)"
    manifest_prefill_ffn_overlap="$(sed -n 's/^prefill_ffn_overlap=//p' "${manifest}" | head -n 1)"
    source_digest="$(sed -n 's/^profile_source_sha256=//p' "${manifest}" | head -n 1)"
    flags_digest="$(sed -n 's/^profile_cflags_sha256=//p' "${manifest}" | head -n 1)"
    xo_digest="$(sed -n 's/^xo_sha256=//p' "${manifest}" | head -n 1)"
    [[ "${profile}" == "${QUANTIZED_LAYER_PROFILE}" ]] || {
        quantized_layer_profile_die \
            "XO profile mismatch for ${xo}: manifest=${profile:-missing} requested=${QUANTIZED_LAYER_PROFILE}"
        return 2
    }
    if [[ -z "${manifest_silu_lanes}" ]]; then
        if [[ "${QUANTIZED_LAYER_SILU_LANES_REQUESTED:-0}" == 1 ||
              "${expected_silu_lanes}" != 1 ]]; then
            quantized_layer_profile_die \
                "XO SiLU lane identity missing for explicit lanes=${expected_silu_lanes}: ${xo}"
            return 2
        fi
    elif [[ "${manifest_silu_lanes}" != "${expected_silu_lanes}" ]]; then
        quantized_layer_profile_die \
            "XO SiLU lane mismatch for ${xo}: manifest=${manifest_silu_lanes} requested=${expected_silu_lanes}"
        return 2
    fi
    if [[ -z "${manifest_rms_lanes}" ]]; then
        if [[ "${QUANTIZED_LAYER_RMS_LANES_REQUESTED:-0}" == 1 ||
              "${expected_rms_lanes}" != "${default_rms_lanes}" ]]; then
            quantized_layer_profile_die \
                "XO RMS lane identity missing for explicit lanes=${expected_rms_lanes}: ${xo}"
            return 2
        fi
    elif [[ "${manifest_rms_lanes}" != "${expected_rms_lanes}" ]]; then
        quantized_layer_profile_die \
            "XO RMS lane mismatch for ${xo}: manifest=${manifest_rms_lanes} requested=${expected_rms_lanes}"
        return 2
    fi
    if [[ -z "${manifest_weight_outstanding}" ]]; then
        if [[ "${QUANTIZED_LAYER_WEIGHT_READ_OUTSTANDING_REQUESTED:-0}" == 1 ||
              "${expected_weight_outstanding}" != "${default_weight_outstanding}" ]]; then
            quantized_layer_profile_die \
                "XO weight-read identity missing for explicit outstanding=${expected_weight_outstanding}: ${xo}"
            return 2
        fi
    elif [[ "${manifest_weight_outstanding}" != "${expected_weight_outstanding}" ]]; then
        quantized_layer_profile_die \
            "XO weight-read outstanding mismatch for ${xo}: manifest=${manifest_weight_outstanding} requested=${expected_weight_outstanding}"
        return 2
    fi
    if [[ -z "${manifest_attention_wave}" ]]; then
        if [[ "${QUANTIZED_LAYER_ATTENTION_WAVE_REQUESTED:-0}" == 1 ||
              "${expected_attention_wave}" != "${default_attention_wave}" ]]; then
            quantized_layer_profile_die \
                "XO attention-wave identity missing for explicit wave=${expected_attention_wave}: ${xo}"
            return 2
        fi
    elif [[ "${manifest_attention_wave}" != "${expected_attention_wave}" ]]; then
        quantized_layer_profile_die \
            "XO attention wave mismatch for ${xo}: manifest=${manifest_attention_wave} requested=${expected_attention_wave}"
        return 2
    fi
    if [[ -z "${manifest_prefill_ffn_overlap}" ]]; then
        if [[ "${QUANTIZED_LAYER_PREFILL_FFN_OVERLAP_REQUESTED:-0}" == 1 ||
              "${expected_prefill_ffn_overlap}" != "${default_prefill_ffn_overlap}" ]]; then
            quantized_layer_profile_die \
                "XO Prefill FFN overlap identity missing for explicit overlap=${expected_prefill_ffn_overlap}: ${xo}"
            return 2
        fi
    elif [[ "${manifest_prefill_ffn_overlap}" != "${expected_prefill_ffn_overlap}" ]]; then
        quantized_layer_profile_die \
            "XO Prefill FFN overlap mismatch for ${xo}: manifest=${manifest_prefill_ffn_overlap} requested=${expected_prefill_ffn_overlap}"
        return 2
    fi
    [[ "${source_digest}" == "${QUANTIZED_LAYER_PROFILE_SOURCE_SHA256}" ]] || {
        quantized_layer_profile_die "profile helper source changed for ${xo}; rebuild it"
        return 2
    }
    [[ "${flags_digest}" == "${QUANTIZED_LAYER_PROFILE_CFLAGS_SHA256}" ]] || {
        quantized_layer_profile_die "profile flags changed for ${xo}; rebuild it"
        return 2
    }
    if [[ -s "${xo}" ]]; then
        [[ "${xo_digest}" == "$(sha256sum "${xo}" | awk '{print $1}')" ]] || {
            quantized_layer_profile_die "XO bytes do not match its profile manifest: ${xo}"
            return 2
        }
    else
        quantized_layer_profile_die "profile manifest has no XO: ${xo}"
        return 2
    fi
}

quantized_layer_profile_write_xo_manifest() {
    local precision="$1" xo="$2" manifest
    manifest="$(quantized_layer_profile_manifest_for_xo "${xo}")"
    [[ -s "${xo}" ]] || {
        quantized_layer_profile_die "cannot manifest missing XO: ${xo}"
        return 2
    }
    {
        printf 'schema=quantized_layer_profile_v1\n'
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
        printf 'xo_sha256=%s\n' "$(sha256sum "${xo}" | awk '{print $1}')"
    } >"${manifest}"
}
