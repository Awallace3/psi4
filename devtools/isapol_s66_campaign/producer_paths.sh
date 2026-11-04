#!/bin/bash
# Closed path selection only; no execution, credentials, overrides or submission.
ccasp_select_paths() {
    [[ $# -eq 1 ]] || return 2
    local label=$1 root=/storage/project/r-cs207-0/awallace43/runs
    case "$label" in
        cb63305f41|ed561666fc)
            CCASP_QUALIFICATION="$root/ccasp-native-s66-$label-checks001/payload/installed-validation.json"
            ;;
        grid400-1aaa9e0a30)
            CCASP_QUALIFICATION="$root/ccasp-native-s66-$label-installed001/payload/installed-validation.json"
            ;;
        *) return 2 ;;
    esac
    CCASP_BUILD="$root/ccasp-native-s66-$label-build001"
    CCASP_INPUTS="$root/ccasp-native-s66-$label-preparation001"
    CCASP_DESCRIPTORS="$root/ccasp-native-s66-$label-descriptors001/payload"
}
