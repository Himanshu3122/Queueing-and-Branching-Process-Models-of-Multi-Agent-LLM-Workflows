"""Download the public datasets into ``data/raw`` and verify their SHA-256 checksums.

The data files are not redistributed with the repository (see the licences in ``docs/real_data.md``).
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

_HF = "https://huggingface.co/datasets/zai-org/AgentInstruct/resolve/main/data/"
_AZ = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/"
_MC = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/"
_BF = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/possible_answer/"

DATASETS: dict[str, tuple[str, str]] = {
    # local path (relative to data/raw): (url, sha256)
    "agentinstruct/os.parquet": (
        _HF + "os-00000-of-00001-971539c34fcc7500.parquet",
        "ea85d61d91ae5c07669293242634e08b577d13d775cee74520b511661c00a241",
    ),
    "agentinstruct/db.parquet": (
        _HF + "db-00000-of-00001-916a87c4725da8c0.parquet",
        "d94980f9e7223066a794ff2bfcc2fc7145287ea762bc572f29ce072cd94441fb",
    ),
    "agentinstruct/alfworld.parquet": (
        _HF + "alfworld-00000-of-00001-302ad687bb3817a4.parquet",
        "d272314940a75333b65885cb9f6e7427340f63ddb15adf2242b2f82bf32c1dd6",
    ),
    "agentinstruct/webshop.parquet": (
        _HF + "webshop-00000-of-00001-9f2ae60445e11b4e.parquet",
        "ddbb00b189191b5d77a272f67e879d4cd8dbe22b404bcf60f1e339cdc3947f77",
    ),
    "agentinstruct/kg.parquet": (
        _HF + "kg-00000-of-00001-9e159f6d0557d229.parquet",
        "2fa0a5e45bc3777377e21c5ee8a08c4b6c277ab475e5906bb491a028f9806b84",
    ),
    "azure/AzureLLMInferenceTrace_conv.csv": (
        _AZ + "AzureLLMInferenceTrace_conv.csv",
        "2f1e5b666d4e3055fdbba98598ce2ec307767b9064e03e2fa46676dbcc7d0bf8",
    ),
    # Mooncake (FAST 2025): real one-hour traces of Kimi's tool/agent and conversation workloads
    "mooncake/toolagent_trace.jsonl": (
        _MC + "toolagent_trace.jsonl",
        "48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71",
    ),
    "mooncake/conversation_trace.jsonl": (
        _MC + "conversation_trace.jsonl",
        "b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df",
    ),
    # BFCL (ICML 2025): ground-truth calls per query -> real tool fan-out law
    "bfcl/possible_answer/BFCL_v3_live_simple.json": (
        _BF + "BFCL_v3_live_simple.json",
        "4a1988780cdd8a723967587614cb84a9f53aa5d6fd37f83c78a0dd7acc2f3d15",
    ),
    "bfcl/possible_answer/BFCL_v3_live_multiple.json": (
        _BF + "BFCL_v3_live_multiple.json",
        "97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084",
    ),
    "bfcl/possible_answer/BFCL_v3_live_parallel.json": (
        _BF + "BFCL_v3_live_parallel.json",
        "5e3d26c9103dd25912003d3c5f681f30efc6fd2b5f8ba474450f2df21fdf6fcb",
    ),
    "bfcl/possible_answer/BFCL_v3_live_parallel_multiple.json": (
        _BF + "BFCL_v3_live_parallel_multiple.json",
        "f5b5f360556c5feb51db46fb9f56ee4b304f4b45b161599bbb14161c98a2873f",
    ),
    "bfcl/possible_answer/BFCL_v3_parallel.json": (
        _BF + "BFCL_v3_parallel.json",
        "3af54bf04e5b7640b03a9b6a82440f52474feb24b398f94435fd2b155c7727de",
    ),
    "bfcl/possible_answer/BFCL_v3_parallel_multiple.json": (
        _BF + "BFCL_v3_parallel_multiple.json",
        "df4d2c16238c1465dc04697c273cf3ab4a450075fcd274e3dc86ad5bf0049b75",
    ),
}

_SG = "https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories/resolve/baf3a4e4bff514d48ddc08a93a2ade5c126212c7/data/"
DATASETS.update(
    {  # SWE-Gym (ICML 2025): sampled OpenHands trajectories with both outcomes (301 MB)
        f"swegym/train.raw-0000{i}-of-00003.parquet": (_SG + f"train.raw-0000{i}-of-00003.parquet", digest)
        for i, digest in enumerate(
            (
                "29d038b0c34eb0e50a19730f3cf3daa5a156c2d7e9c871c01b568acbab6aae07",
                "64a5d435b777a47e2544de1cc3bf61e5452bab1237fc43e70220e13f2f901516",
                "1e388384f0dbf65421b53ef37d4f92c10e1e458f597bf61326fa047b8e21b597",
            )
        )
    }
)

_BG = "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/"
DATASETS.update(
    {  # BurstGPT (KDD 2025): part 1 (61 days) and part 3 (110 days, with latency and sessions)
        "burstgpt/BurstGPT_1.csv": (
            _BG + "BurstGPT_1.csv",
            "4bb3783693d0a435686fbfc885615d2349bd067239079fa4b749f2e679e12122",
        ),
        "burstgpt/BurstGPT_3.csv": (
            _BG + "BurstGPT_3.csv",
            "2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f",
        ),
    }
)

DATASETS["mast/MAD_full_dataset.json"] = (  # MAST (NeurIPS 2025): 1,642 multi-agent runs, 200 MB
    "https://huggingface.co/datasets/mcemri/MAST-Data/resolve/95118ac951421753cf1deb87ddea3b01e693c41b/MAD_full_dataset.json",
    "d636ac63dfc1c6af2d312e862f4b7d383b62d3898d431ccd0e7a79d21d85406f",
)

_AZ24 = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
DATASETS.update(
    {  # Azure LLM inference trace 2024 (HPCA 2025): one week of a code and a conversation service, 1.8 GB
        "azure2024/AzureLLMInferenceTrace_code_1week.csv": (
            _AZ24 + "AzureLLMInferenceTrace_code_1week.csv",
            "71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f",
        ),
        "azure2024/AzureLLMInferenceTrace_conv_1week.csv": (
            _AZ24 + "AzureLLMInferenceTrace_conv_1week.csv",
            "a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941",
        ),
    }
)

ARCHIVES: dict[str, tuple[str, str, str]] = {
    # local zip path: (url pinned to a commit, sha256, top-level folder to extract next to it)
    "whowhen/agents_failure_attribution.zip": (
        "https://codeload.github.com/ag2ai/Agents_Failure_Attribution/zip/1bcbbd96ba563400cededae6189833b8ea59dcae",
        "ee9ff8801a68a2becc0cc85255c2d5eee25c7b5a8fb9a9586982305a9850f1a9",
        "Who&When",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_datasets(root: str | Path = "data/raw", force: bool = False) -> list[Path]:
    """Download missing files (or all with ``force``) and verify every checksum."""
    root = Path(root)
    paths = []
    for rel, (url, digest) in DATASETS.items():
        path = root / rel
        if force or not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading %s", url)
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs above
            tmp.replace(path)
        actual = _sha256(path)
        if actual != digest:
            raise ValueError(f"checksum mismatch for {path}: expected {digest}, got {actual}")
        paths.append(path)
    for rel, (url, digest, member) in ARCHIVES.items():
        archive = root / rel
        if force or not archive.is_file():
            archive.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading %s", url)
            tmp = archive.with_suffix(archive.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs above
            tmp.replace(archive)
        actual = _sha256(archive)
        if actual != digest:
            raise ValueError(f"checksum mismatch for {archive}: expected {digest}, got {actual}")
        target = archive.parent / member
        if force or not target.is_dir():
            _extract_member(archive, member, archive.parent)
        paths.append(target)
    return paths


def _extract_member(archive: Path, member: str, dest: Path) -> None:
    """Extract ``<top>/<member>/...`` of a GitHub zip to ``dest/<member>/...`` (no path traversal)."""
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if len(parts) < 2 or parts[1] != member or info.is_dir():
                continue
            rel = Path(*parts[1:])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe path in archive: {info.filename}")
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(info))
