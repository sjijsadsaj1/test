"""Utilities for selecting an explicit, reproducible Habitat episode subset."""

import json
import os
from typing import Any, Iterable, List, Mapping, Tuple


def scene_key(scene_id: str) -> str:
    """Return the HM3D/MP3D scene key from a Habitat scene path."""
    return os.path.basename(str(scene_id)).split(".", 1)[0]


def episode_key(episode: Any) -> Tuple[str, str, str]:
    """Build the key needed to distinguish ObjectNav episodes reliably."""
    return (
        scene_key(episode.scene_id),
        str(episode.episode_id),
        str(episode.object_category),
    )


def select_episodes_from_manifest(
    episodes: Iterable[Any], manifest_path: str
) -> List[Any]:
    """Select and order Habitat episodes using a JSON manifest.

    Episode IDs repeat across scenes and target categories in Habitat ObjectNav,
    so a manifest entry is identified by ``scene_key``, ``episode_id`` and
    ``object_category`` together.
    """
    manifest_path = os.path.abspath(os.path.expanduser(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    entries = manifest.get("episodes") if isinstance(manifest, Mapping) else manifest
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Episode manifest must contain a non-empty 'episodes' list: {manifest_path}"
        )

    episode_by_key = {}
    for episode in episodes:
        key = episode_key(episode)
        if key in episode_by_key:
            raise ValueError(f"Dataset contains a duplicate episode key: {key}")
        episode_by_key[key] = episode

    selected = []
    requested_keys = set()
    missing = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Manifest episode {index} must be a JSON object")
        raw_scene = entry.get("scene_key", entry.get("scene_id", ""))
        key = (
            scene_key(str(raw_scene)),
            str(entry.get("episode_id", "")),
            str(entry.get("object_category", "")),
        )
        if not all(key):
            raise ValueError(
                "Manifest episode %d needs scene_key (or scene_id), episode_id, "
                "and object_category" % index
            )
        if key in requested_keys:
            raise ValueError(f"Manifest contains a duplicate episode key: {key}")
        requested_keys.add(key)

        episode = episode_by_key.get(key)
        if episode is None:
            missing.append(key)
        else:
            selected.append(episode)

    if missing:
        preview = ", ".join(repr(key) for key in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(
            f"{len(missing)} manifest episodes were not found in the dataset: "
            f"{preview}{suffix}"
        )

    return selected
