from __future__ import annotations

import os
import datajoint as dj
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from element_interface.utils import dict_to_uuid, find_full_path

from element_array_ephys import ephys
from element_array_ephys.spike_sorting import si_spike_sorting as ephys_sorter


logger = dj.logger

schema = dj.schema()


def activate(
    schema_name,
    *,
    create_schema=True,
    create_tables=True,
):
    """
    activate(schema_name, *, create_schema=True, create_tables=True)
        :param schema_name: schema name on the database server to activate the `ephys_curation` schema
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
    """

    if not ephys.schema.is_activated():
        raise RuntimeError("Please activate the `ephys` schema first.")
    if not ephys_sorter.schema.is_activated():
        raise RuntimeError("Please activate the `si_spike_sorting` schema first.")

    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects={**ephys.__dict__, **ephys_sorter.__dict__},
    )
    try:
        ephys.ClusterQualityLabel.insert1(
            ("no kilosort label", "unsorted spikes (Phy default)"), skip_duplicates=True
        )
    except Exception as e:
        logger.warning(f"Failed to insert default ClusterQualityLabel: {e}")


@schema
class CurationMethod(dj.Lookup):
    definition = """
    # Curation method
    curation_method: varchar(16)  # method/package used to perform manual curation (e.g. Phy, FigURL, etc.)
    """
    contents = [
        ("Phy",),
    ]


@schema
class ManualCuration(dj.Manual):
    definition = """
    # Manual curation from an ephys.Clustering
    -> ephys.Clustering
    curation_id: int
    ---
    curation_datetime: datetime    # UTC time when the curation was performed
    parent_curation_id=-1: int     # if -1, this curation is based on the raw spike sorting results
    -> CurationMethod              # which method/package used for manual curation (inform how to ingest the results)
    description="": varchar(1000)  # user-defined description/note of the curation
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)
        ---
        file: filepath@ephys-processed
        """

    @classmethod
    def prepare_manual_curation(
        cls,
        key,
        *,
        parent_curation_id=-1,
        curation_method="Phy",
        if_exists="skip",
        download_binary=False,
    ):
        """
        Create a new directory to for a new round of manual curation
        Download the spike sorting results for new manual curation from the specified "key" and "parent_curation_id".
        Store the initial meta information in json file.
        Args:
            key (dict): PK of the ephys.Clustering table
            parent_curation_id (int): curation_id to be used as the starting point for this new curation
            curation_method (str): method/package used for manual curation (e.g. Phy)
            if_exists (str): overwrite|skip
            download_binary (bool): if True, also download the raw ephys (.dat) file

        Returns:
            directory where the spike sorting results are downloaded
        """
        assert if_exists in [
            "overwrite",
            "skip",
        ], f"Invalid if_exists: {if_exists}"

        if curation_method != "Phy":
            raise ValueError(f"Unsupported curation method: {curation_method}")

        init_datetime = datetime.now(timezone.utc)

        # Download the spike sorting results
        assert ephys.CuratedClustering & key, f"Invalid ephys.Clustering key: {key}"
        if len(ephys.CuratedClustering.Unit & key) == 0:
            logger.warning("This clustering has no units!!!")

        if parent_curation_id == -1:
            assert (
                ephys_sorter.SIExport & key
            ), "SIExport not found for the specified key"
            files_query = (
                ephys_sorter.SIExport.File
                & key
                & "file_name LIKE 'phy%' AND file_name NOT LIKE '%recording.dat'"
            )
        else:
            assert cls & {
                **key,
                "curation_id": parent_curation_id,
            }, "ManualCuration not found for the specified key"
            files_query = cls.File & {**key, "curation_id": parent_curation_id}

        # create new directory for new curation
        output_dir = (ephys.ClusteringTask & key).fetch1("clustering_output_dir")
        output_dir = (
            Path(ephys.get_processed_root_data_dir()) / output_dir / "curations"
        )

        dirname = f"{curation_method}_parentID_" + (
            "orig" if parent_curation_id == -1 else f"{parent_curation_id}"
        )
        curation_output_dir = output_dir / dirname
        curation_output_dir.mkdir(parents=True, exist_ok=True)

        # download spike sorting files and copy to the new directory
        logger.info(
            f"New manual curation: {curation_output_dir} - Downloading {len(files_query)} files..."
        )

        for f in files_query.fetch("file"):
            f = Path(f)
            if f.name.startswith(".") and f.suffix in (".json", ".pickle"):
                continue
            new_f = curation_output_dir / f.name
            if not new_f.exists() or if_exists == "overwrite":
                shutil.copy2(f, new_f)

        if download_binary:
            new_f = curation_output_dir / "recording.dat"
            if not new_f.exists() or if_exists == "overwrite":
                raw_file_query = (
                    ephys_sorter.SIExport.File & key & "file_name = 'phy/recording.dat'"
                )
                if raw_file_query:
                    logger.info("Downloading raw ephys data file...")
                    f = Path(raw_file_query.fetch1("file"))
                    shutil.copy2(f, new_f)
                else:
                    logger.warning("Raw ephys data file not found.")

        # write "entry" into a json file
        with open(curation_output_dir / ".manual_curation_entry.json", "w") as f:
            json.dump(
                {
                    **key,
                    "parent_curation_id": parent_curation_id,
                    "curation_method": curation_method,
                },
                f,
                default=str,
            )

        return curation_output_dir

    @classmethod
    def insert_manual_curation(
        cls,
        curation_output_dir,
        *,
        key=None,
        parent_curation_id=None,
        curation_method="Phy",
        description="",
        delete_local_dir=True,
    ):
        """
        Insert a new manual curation into the database
        1. Get a new "curation_id" (auto-incremented)
        2. Copy the curation_output_dir to a new directory with the new curation_id
        3. Insert the new curation into the database (excluding the raw recording.dat file)
        4. Delete the old curation directory
        5. Optionally delete the new curation directory
        Args:
            curation_output_dir: directory where the curation results are stored
            key: ephys.Clustering key
            parent_curation_id: curation_id of the parent curation
            curation_method: method/package used for manual curation (e.g. Phy)
            description: user-defined description/note of the curation
            delete_local_dir: if True, delete the new curation directory after inserting the curation into the database

        Returns: `key` of the newly inserted manual curation
        """

        curation_output_dir = Path(curation_output_dir)

        # Light logic to safeguard against re-inserting the same curation
        if (curation_output_dir / ".manual_curation_entry.json").exists():
            if key is not None:
                logger.warning(
                    ".manual_curation_entry.json already exists. Ignoring inputs."
                )

            with open(curation_output_dir / ".manual_curation_entry.json", "r") as f:
                key = json.load(f)

            parent_curation_id = key.pop("parent_curation_id")
            curation_method = key.pop("curation_method")

            if "curation_id" in key:
                print(f"Manual curation already inserted: {key}")
                if (
                    dj.utils.user_choice("Insert a new manual curation anyway?")
                    != "yes"
                ):
                    print("Canceled")
                    return
        else:
            if parent_curation_id is None or key is None:
                raise ValueError(
                    ".manual_curation_entry.json not found. `key` AND `parent_curation_id` must be specified"
                )

        if curation_method != "Phy":
            raise ValueError(f"Unsupported curation method: {curation_method}")

        from element_array_ephys.readers import kilosort

        kilosort_dataset = kilosort.Kilosort(curation_output_dir)
        assert (
            kilosort_dataset.data
        ), f"Invalid Phy output directory: {curation_output_dir}"

        curate_datetime = datetime.now(timezone.utc)
        curation_id = (
            ephys.Clustering.aggr(cls, count="count(curation_id)", keep_all_rows=True)
            & key
        ).fetch1("count") + 1
        logger.info(f"New curation id: {curation_id}")

        key["curation_id"] = curation_id
        with cls.connection.transaction:
            entry = {
                **key,
                "curation_datetime": curate_datetime,
                "parent_curation_id": parent_curation_id,
                "curation_method": curation_method,
                "description": description,
            }
            cls.insert1(entry)

            # rename curation_output_dir folder into curation_id (skip the raw recording.dat file)
            new_curation_output_dir = (
                curation_output_dir.parent
                / f"{curation_method}_curationID_{curation_id}"
            )
            new_curation_output_dir.mkdir(parents=True, exist_ok=True)
            for f in curation_output_dir.glob("*"):
                if f.is_file() and f.name != "recording.dat":
                    shutil.copy2(
                        f, new_curation_output_dir / f.relative_to(curation_output_dir)
                    )
                elif f.is_dir():
                    shutil.copytree(
                        f, new_curation_output_dir / f.relative_to(curation_output_dir)
                    )

            logger.info(f"Inserting files from {new_curation_output_dir}...")
            cls.File.insert(
                [
                    {
                        **key,
                        "file_name": f.relative_to(new_curation_output_dir).as_posix(),
                        "file": f,
                    }
                    for f in new_curation_output_dir.rglob("*")
                    if f.is_file()
                    and f.name not in ("recording.dat", ".manual_curation_entry.json")
                ]
            )

        logger.info(f"New manual curation successfully inserted: {key}")

        # write "entry" into a json file
        with open(new_curation_output_dir / ".manual_curation_entry.json", "w") as f:
            json.dump(entry, f, default=str)

        # Delete the old curation directory
        try:
            shutil.rmtree(curation_output_dir)
        except Exception as e:
            logger.error(
                f"Failed to fully delete the old curation directory (please try to delete manually):\n\t{curation_output_dir}\n{e}"
            )

        if delete_local_dir:
            try:
                shutil.rmtree(new_curation_output_dir)
            except Exception as e:
                logger.error(
                    f"Failed to fully delete the new curation directory (please try to delete manually):\n\t{new_curation_output_dir}\n{e}"
                )

        return key


@schema
class OfficialCuration(dj.Manual):
    definition = """
    -> ephys.Clustering
    ---
    -> ManualCuration
    """


@schema
class ApplyOfficialCuration(dj.Imported):
    definition = """
    -> OfficialCuration
    ---
    execution_time: datetime        # datetime of the start of this step
    new_unit_count: int             # number of new units added
    removed_unit_count: int         # number of units removed
    """

    @property
    def key_source(self):
        return OfficialCuration & ephys.CuratedClustering

    def make(self, key):
        """
        High level logic
        Step 1: delete units from ephys.CuratedClustering.Unit that are not in the new curation (merged or split)
        Step 2: add new entries for new units (newly merged or split)

        Note: when replacing an OfficialCuration, manual steps must be taken
        for a reset ingestion of ephys.CuratedClustering and below
          - delete: (ephys.CuratedClustering & key).delete()
          - repopulate: calls populate for `CuratedClustering`, `WaveformSet`, `QualityMetrics`
        """
        from element_array_ephys.readers import kilosort

        # Resolve full key including curation_id from OfficialCuration
        # (key only contains Clustering PK; curation_id is a dependent attribute)
        official_key = (OfficialCuration & key).fetch1()

        curated_files = (ManualCuration.File & official_key).fetch("file")
        curation_output_dir = (
            next(Path(f) for f in curated_files if Path(f).name == "params.py")
        ).parent

        curation_method = (ManualCuration & official_key).fetch1(
            "curation_method"
        )

        if curation_method != "Phy":
            raise ValueError(f"Unsupported curation method: {curation_method}")

        clus_key = (ephys.Clustering & key).fetch1("KEY")

        kilosort_dataset = kilosort.Kilosort(curation_output_dir)

        orig_si_unit_map = {
            u: i
            for i, u in enumerate(
                (ephys.CuratedClustering.Unit & clus_key).fetch("unit", order_by="unit")
            )
        }

        new_si_unit_map = pd.read_csv(
            curation_output_dir / "cluster_si_unit_id.tsv", sep="\t", index_col=1
        ).to_dict()["cluster_id"]

        # find set of units that are in the original curation but not in the new
        removed_units = set(orig_si_unit_map) - set(new_si_unit_map)
        # find set of units that are in the new curation but not in the original
        new_units = set(kilosort_dataset.data["cluster_ids"]) - set(
            orig_si_unit_map.values()
        )
        new_si_unit_map.update(
            {i + max(orig_si_unit_map) + 1: u for i, u in enumerate(new_units)}
        )
        new_si_unit_reverse_map = {v: k for k, v in new_si_unit_map.items()}

        # Get channel and electrode-site mapping
        # For SI-processed data, Kilosort's spike_sites contains device_channel_indices
        # (sequential 0, 1, 2...) which may differ from the original channel_idx in
        # EphysRecording.Channel (e.g., 32-63 for multi-insertion probes like MBA).
        # Use the sorting_analyzer's probe to correctly map channel indices to electrodes.
        electrode_query = (ephys.EphysRecording.Channel & clus_key).proj(
            ..., "-channel_name"
        )

        # Get sorting_analyzer to access the probe's channel mapping
        clustering_method, output_dir = (
            ephys.ClusteringTask * ephys.ClusteringParamSet & clus_key
        ).fetch1("clustering_method", "clustering_output_dir")
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        sorter_name = clustering_method.replace(".", "_")
        si_sorting_analyzer_dir = output_dir / sorter_name / "sorting_analyzer"

        # If sorting_analyzer not available locally, fetch from external storage (S3)
        if not si_sorting_analyzer_dir.exists():
            _ = (ephys_sorter.PostProcessing.File & clus_key).fetch("file")

        if si_sorting_analyzer_dir.exists():
            import spikeinterface as si

            sorting_analyzer = si.load_sorting_analyzer(folder=si_sorting_analyzer_dir)
            # Create electrode_map keyed by electrode ID
            electrode_map: dict[int, dict] = {}
            for elec in electrode_query.fetch(as_dict=True):
                elec.pop("channel_idx")
                electrode_map[elec["electrode"]] = elec
            # Map device_channel_indices to electrode info using probe's contact_ids
            channel2electrode_map: dict[int, dict] = {
                chn_idx: electrode_map[int(elec_id)]
                for chn_idx, elec_id in zip(
                    sorting_analyzer.get_probe().device_channel_indices,
                    sorting_analyzer.get_probe().contact_ids,
                )
            }
        else:
            # Fallback for non-SI processed data - use original channel_idx
            channel2electrode_map: dict[int, dict] = {
                chn["channel_idx"]: chn for chn in electrode_query.fetch(as_dict=True)
            }

        sample_rate = kilosort_dataset.data["params"]["sample_rate"]
        spike_times = kilosort_dataset.data["spike_times"]
        kilosort_dataset.extract_spike_depths()
        # -- Spike-sites and Spike-depths --
        spike_sites = np.array(
            [
                channel2electrode_map[s]["electrode"]
                for s in kilosort_dataset.data["spike_sites"]
            ]
        )
        spike_depths = kilosort_dataset.data["spike_depths"]

        # -- Remove units
        with dj.config(safemode=False):
            (
                ephys.CuratedClustering.Unit
                & clus_key
                & [{"unit": u} for u in removed_units]
            ).delete(force=True)

        # -- Insert unit, label, peak-chn
        for cluster_id, cluster_group in zip(
            kilosort_dataset.data["cluster_ids"],
            kilosort_dataset.data["cluster_groups"],
        ):
            unit_key = {**clus_key, "unit": new_si_unit_reverse_map[cluster_id]}
            if cluster_id in new_units:
                # add new unit entry
                unit_channel, _ = kilosort_dataset.get_best_channel(cluster_id)
                unit_spike_times = (
                    spike_times[kilosort_dataset.data["spike_clusters"] == cluster_id]
                    / sample_rate
                )
                spike_count = len(unit_spike_times)

                ephys.CuratedClustering.Unit.insert1(
                    {
                        **unit_key,
                        "cluster_quality_label": "no kilosort label",  # new units will have "no kilosort label" label
                        **channel2electrode_map[unit_channel],
                        "spike_times": unit_spike_times,
                        "spike_count": spike_count,
                        "spike_sites": spike_sites[
                            kilosort_dataset.data["spike_clusters"] == cluster_id
                        ],
                        "spike_depths": spike_depths[
                            kilosort_dataset.data["spike_clusters"] == cluster_id
                        ],
                    },
                    allow_direct_insert=True,
                )

            # insert the new unit label
            ephys.CuratedClustering.ManualLabel.insert1(
                {**unit_key, "manual_label": cluster_group}, allow_direct_insert=True
            )

        self.insert1(
            {
                **key,
                "execution_time": datetime.now(timezone.utc),
                "new_unit_count": len(new_units),
                "removed_unit_count": len(removed_units),
            }
        )


def launch_phy(key, parent_curation_id=-1, download_binary=False):
    """
    Select a spike sorting key for manual curation
    1. download the spike sorting results
    2. launch phy
    3. commit new curation locally
    4. insert new curation into the database
    Args:
        key: ephys.Clustering key
        parent_curation_id: if -1, this curation is based on the raw spike sorting results
        download_raw: if True, also download the raw ephys (.dat) file
        do_insert: if True, insert the new curation into the database (upload result files)
    """
    from phy.apps.template import template_gui

    curation_output_dir = ManualCuration.prepare_manual_curation(
        key, parent_curation_id=parent_curation_id, download_binary=download_binary
    )

    template_gui(curation_output_dir / "params.py")

    description = input("Curation description: ")

    ManualCuration.insert_manual_curation(
        curation_output_dir,
        description=description,
    )
