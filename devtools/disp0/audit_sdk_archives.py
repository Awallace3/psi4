"""LOCAL read-only BOTH pinned archives; no filesystem extraction/runtime use."""

import json
import signal
import urllib.request

import provision_sdk_inputs as inputs

HEADER_SHAS = {
    "cusparse.h": "76c4ca9c08a101291c987776a42c03fa436202ac9fe970a02280452c8d1ee222",
    "cusparse_v2.h": "8e41f603d85873e4c4174bee43f4aeadb84f3441e460650846ec4a5e1140aa7c",
    "nvJitLink.h": "d1f43e48b34cb8bdc2ac32c192e60b1fec882c09c5dae691cb876aae4eafb48e"}


def audit():
    for item in inputs.PACKAGES:
        with urllib.request.urlopen(inputs.package_url(item), timeout=15) as response:
            data = inputs.limited_read(response, item["size"])
        headers, record = inputs.read_package(data, item, inputs.decoder())
        for name, observed in record["headers"].items():
            inputs.contract.require(HEADER_SHAS.get(name) == observed, "Real archive header SHA drift")
        expected_names = {"cusparse.h", "cusparse_v2.h"} if item["family"] == "cusparse" else {"nvJitLink.h"}
        inputs.contract.require(set(headers) == expected_names, "Real archive header inventory changed")
        print(json.dumps({
            "scope": "LOCAL_REAL_ARCHIVE_REGRESSION_ONLY",
            "url": record["url"], "archive_bytes": len(data),
            "verified_archive_sha256": item["sha256"],
            "index": record["index"], "headers": record["headers"],
            "developer_link_graph": record["developer_link_graph"],
            "external_expected_binding": record["external_expected_binding"],
            "binding_provenance": inputs.COMPONENT_BINDING_MANIFEST,
            "actual_external_file_reverified_here": False,
            "helper_sha256": inputs.contract.digest(inputs.__file__),
            "no_files_extracted": True, "no_provisioning_or_allocation": True,
            "sdk_or_e2_admission": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("60s local audit bound")))
    signal.alarm(60)
    audit()
