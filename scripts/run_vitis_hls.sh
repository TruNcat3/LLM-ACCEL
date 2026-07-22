#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <tcl-script>" >&2
    exit 2
fi

cd "$(dirname "$0")/.."

tcl_script="$1"
if [ ! -f "$tcl_script" ]; then
    echo "HLS launcher: TCL script not found: $tcl_script" >&2
    exit 2
fi

min_available_gib="${HLS_MIN_AVAILABLE_GIB:-50}"
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
required_kib=$((min_available_gib * 1024 * 1024))

if [ "$available_kib" -lt "$required_kib" ]; then
    available_gib="$(awk -v kib="$available_kib" 'BEGIN { printf "%.1f", kib / 1024 / 1024 }')"
    echo "HLS memory guard: MemAvailable=${available_gib}GiB, need at least ${min_available_gib}GiB; not starting vitis_hls." >&2
    echo "Set HLS_MIN_AVAILABLE_GIB to adjust this threshold." >&2
    exit 75
fi

project_name="${LLM_FPGA_HLS_PROJECT_NAME:-}"
if [ -z "$project_name" ]; then
    project_name="$(
        awk '
            $1 == "open_project" {
                for (i = 2; i <= NF; i++) {
                    if ($i == "-reset") {
                        continue
                    }
                    print $i
                    exit
                }
            }
            $1 == "set" && $2 == "project_name" {
                print $3
                exit
            }
        ' "$tcl_script" | tr -d '{}"'
    )"
fi

if [ -n "$project_name" ] && [ -d "$project_name/solution1" ]; then
    archive_needed=false
    for rel_path in solution1/syn/report solution1/sim/report; do
        if [ -d "$project_name/$rel_path" ] && find "$project_name/$rel_path" -type f -print -quit | grep -q .; then
            archive_needed=true
        fi
    done
    for rel_path in solution1/solution1.log solution1/.autopilot/db/autopilot.flow.log; do
        if [ -s "$project_name/$rel_path" ]; then
            archive_needed=true
        fi
    done

    if [ "$archive_needed" = true ]; then
        archive_root="${HLS_REPORT_ARCHIVE_DIR:-reports/hls_report_archive}"
        timestamp="$(date +%Y%m%d_%H%M%S)"
        archive_dir="$archive_root/${project_name}_${timestamp}"
        mkdir -p "$archive_dir"

        for rel_path in solution1/syn/report solution1/sim/report; do
            if [ -d "$project_name/$rel_path" ] && find "$project_name/$rel_path" -type f -print -quit | grep -q .; then
                mkdir -p "$archive_dir/$(dirname "$rel_path")"
                cp -a "$project_name/$rel_path" "$archive_dir/$rel_path"
            fi
        done
        for rel_path in solution1/solution1.log solution1/.autopilot/db/autopilot.flow.log; do
            if [ -s "$project_name/$rel_path" ]; then
                mkdir -p "$archive_dir/$(dirname "$rel_path")"
                cp -a "$project_name/$rel_path" "$archive_dir/$rel_path"
            fi
        done

        {
            printf "project=%s\n" "$project_name"
            printf "tcl=%s\n" "$tcl_script"
            printf "archived_at=%s\n" "$(date -Iseconds)"
            printf "reason=pre-open_project-reset snapshot\n"
        } > "$archive_dir/manifest.txt"
        echo "HLS report archive: saved existing reports for $project_name to $archive_dir" >&2
    fi
fi

exec env -u DEBUG vitis_hls -f "$tcl_script"
