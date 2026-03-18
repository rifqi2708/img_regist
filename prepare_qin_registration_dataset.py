#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import SimpleITK as sitk

SOURCE_ROOT = Path("/workspaces/img_regist/qin_prostate_repeatability")
DATASET_ROOT = Path("/workspaces/img_regist/dataset")
RAW_LINK = DATASET_ROOT / "raw" / "qin_prostate_repeatability"

INCLUDE_KEYWORDS = ["T2", "T2W", "TSE", "AX T2"]
EXCLUDE_KEYWORDS = ["ADC", "DWI", "BVAL", "DCE"]

TAG_STUDY_DATE = "0008|0020"
TAG_SERIES_DESCRIPTION = "0008|103e"
TAG_MODALITY = "0008|0060"
TAG_SEQUENCE_NAME = "0018|0024"
TAG_PIXEL_SPACING = "0028|0030"

METADATA_COLUMNS = [
    "patient_id",
    "fixed_path",
    "moving_path",
    "fixed_series_description",
    "moving_series_description",
    "study_date_fixed",
    "study_date_moving",
]


@dataclass
class MRSeriesMeta:
    patient_id: str
    study_uid: str
    series_uid: str
    series_path: Path
    study_date: str
    series_description: str
    modality: str
    sequence_name: str
    spacing_proxy: str
    slice_count: int


def classify_series_prefix(series_name: str) -> str:
    if series_name.startswith("MR_"):
        return "MR"
    if series_name.startswith("SEG_"):
        return "SEG"
    if series_name.startswith("SR_"):
        return "SR"
    return "OTHER"


def read_dicom_metadata(first_dcm_file: Path) -> Dict[str, str]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(first_dcm_file))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    def get_tag(tag: str) -> str:
        if reader.HasMetaDataKey(tag):
            return reader.GetMetaData(tag).strip()
        return ""

    return {
        "study_date": get_tag(TAG_STUDY_DATE),
        "series_description": get_tag(TAG_SERIES_DESCRIPTION),
        "modality": get_tag(TAG_MODALITY),
        "sequence_name": get_tag(TAG_SEQUENCE_NAME),
        "spacing_proxy": get_tag(TAG_PIXEL_SPACING),
    }


def discover_dataset(source_root: Path, warnings: List[str]) -> Dict[str, Dict]:
    dataset: Dict[str, Dict] = {}
    patient_dirs = sorted(
        p for p in source_root.iterdir() if p.is_dir() and p.name.startswith("PCAMPMRI-")
    )

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        study_dirs = sorted(s for s in patient_dir.iterdir() if s.is_dir())

        patient_entry = {
            "patient_id": patient_id,
            "patient_path": patient_dir,
            "studies": {},
        }

        for study_dir in study_dirs:
            series_dirs = sorted(s for s in study_dir.iterdir() if s.is_dir())
            series_by_type = {"MR": [], "SEG": [], "SR": [], "OTHER": []}
            for series_dir in series_dirs:
                series_type = classify_series_prefix(series_dir.name)
                series_by_type[series_type].append(series_dir)

            patient_entry["studies"][study_dir.name] = {
                "study_uid": study_dir.name,
                "study_path": study_dir,
                "series_by_type": series_by_type,
                "mr_series_meta": [],
            }

        if not study_dirs:
            warnings.append(f"{patient_id}: no study folders found.")

        dataset[patient_id] = patient_entry

    return dataset


def extract_mr_metadata(dataset: Dict[str, Dict], warnings: List[str]) -> List[MRSeriesMeta]:
    all_mr_rows: List[MRSeriesMeta] = []

    for patient_id, patient_entry in dataset.items():
        for study_uid, study_entry in patient_entry["studies"].items():
            mr_series_dirs: List[Path] = study_entry["series_by_type"]["MR"]
            for series_dir in mr_series_dirs:
                dcm_files = sorted(series_dir.glob("*.dcm"))
                if not dcm_files:
                    warnings.append(
                        f"{patient_id} | {study_uid} | {series_dir.name}: no .dcm files in MR series."
                    )
                    continue

                try:
                    tags = read_dicom_metadata(dcm_files[0])
                except Exception as exc:  # pylint: disable=broad-except
                    warnings.append(
                        f"{patient_id} | {study_uid} | {series_dir.name}: failed reading DICOM metadata ({exc})."
                    )
                    continue

                row = MRSeriesMeta(
                    patient_id=patient_id,
                    study_uid=study_uid,
                    series_uid=series_dir.name,
                    series_path=series_dir,
                    study_date=tags["study_date"],
                    series_description=tags["series_description"],
                    modality=tags["modality"],
                    sequence_name=tags["sequence_name"],
                    spacing_proxy=tags["spacing_proxy"],
                    slice_count=len(dcm_files),
                )

                study_entry["mr_series_meta"].append(row)
                all_mr_rows.append(row)

    return all_mr_rows


def choose_t2_series(
    candidates: List[MRSeriesMeta],
) -> Tuple[Optional[MRSeriesMeta], str]:
    if not candidates:
        return None, "No T2 candidate after include/exclude filtering."

    if len(candidates) == 1:
        c = candidates[0]
        return c, f"Single T2 candidate selected ({c.series_uid}, slices={c.slice_count})."

    max_slices = max(c.slice_count for c in candidates)
    top_by_slices = [c for c in candidates if c.slice_count == max_slices]

    if len(top_by_slices) == 1:
        c = top_by_slices[0]
        return (
            c,
            f"Multiple T2 candidates; selected highest slice count ({c.series_uid}, slices={c.slice_count}).",
        )

    spacing_counter = Counter((c.spacing_proxy or "UNKNOWN") for c in candidates)
    modal_spacing = sorted(spacing_counter.items(), key=lambda x: (-x[1], x[0]))[0][0]
    top_by_spacing = [c for c in top_by_slices if (c.spacing_proxy or "UNKNOWN") == modal_spacing]

    if len(top_by_spacing) == 1:
        c = top_by_spacing[0]
        return (
            c,
            "Multiple T2 candidates tied on slice count; "
            f"selected common spacing '{modal_spacing}' ({c.series_uid}).",
        )

    chosen = sorted(top_by_spacing, key=lambda c: c.series_uid)[0]
    return (
        chosen,
        "Multiple T2 candidates tied on slice count and spacing; "
        f"selected lexicographically smallest series UID ({chosen.series_uid}).",
    )


def contains_any(text: str, keywords: List[str]) -> bool:
    upper_text = text.upper()
    return any(keyword.upper() in upper_text for keyword in keywords)


def select_registration_pairs(dataset: Dict[str, Dict], warnings: List[str]) -> Dict[str, Dict]:
    selection: Dict[str, Dict] = {}

    for patient_id, patient_entry in dataset.items():
        studies = patient_entry["studies"]
        patient_result = {
            "patient_id": patient_id,
            "status": "skipped",
            "skip_reason": "",
            "study_dates": {},
            "fixed": None,
            "moving": None,
            "reasoning": "",
        }

        if len(studies) < 2:
            patient_result["skip_reason"] = "Fewer than 2 study folders; cannot form baseline/repeat pair."
            warnings.append(f"{patient_id}: {patient_result['skip_reason']}")
            selection[patient_id] = patient_result
            continue

        study_date_map: Dict[str, str] = {}
        date_conflict = False
        for study_uid, study_entry in studies.items():
            mr_rows: List[MRSeriesMeta] = study_entry["mr_series_meta"]
            dates = sorted({r.study_date for r in mr_rows if r.study_date})
            if not dates:
                warnings.append(f"{patient_id} | {study_uid}: missing StudyDate across MR series.")
                date_conflict = True
                continue
            if len(dates) > 1:
                warnings.append(
                    f"{patient_id} | {study_uid}: conflicting StudyDate values in MR series: {dates}."
                )
                date_conflict = True
                continue
            study_date_map[study_uid] = dates[0]

        patient_result["study_dates"] = study_date_map
        if date_conflict or len(study_date_map) < 2:
            patient_result["skip_reason"] = (
                "Could not establish unique StudyDate for at least two studies."
            )
            warnings.append(f"{patient_id}: {patient_result['skip_reason']}")
            selection[patient_id] = patient_result
            continue

        sorted_studies = sorted(study_date_map.items(), key=lambda x: x[1])
        fixed_study_uid = sorted_studies[0][0]
        moving_study_uid = sorted_studies[-1][0]

        fixed_mr_rows = studies[fixed_study_uid]["mr_series_meta"]
        moving_mr_rows = studies[moving_study_uid]["mr_series_meta"]

        def t2_candidates(rows: List[MRSeriesMeta], study_uid: str) -> List[MRSeriesMeta]:
            valid_rows: List[MRSeriesMeta] = []
            for row in rows:
                if row.modality and row.modality.upper() != "MR":
                    warnings.append(
                        f"{patient_id} | {study_uid} | {row.series_uid}: modality is '{row.modality}', not MR; ignored."
                    )
                    continue
                desc = row.series_description or ""
                include_ok = contains_any(desc, INCLUDE_KEYWORDS)
                exclude_hit = contains_any(desc, EXCLUDE_KEYWORDS)
                if include_ok and not exclude_hit:
                    valid_rows.append(row)
            return valid_rows

        fixed_candidates = t2_candidates(fixed_mr_rows, fixed_study_uid)
        moving_candidates = t2_candidates(moving_mr_rows, moving_study_uid)

        fixed_choice, fixed_reason = choose_t2_series(fixed_candidates)
        moving_choice, moving_reason = choose_t2_series(moving_candidates)

        if fixed_choice is None or moving_choice is None:
            reasons = []
            if fixed_choice is None:
                reasons.append(f"fixed-study ({fixed_study_uid}): {fixed_reason}")
            if moving_choice is None:
                reasons.append(f"moving-study ({moving_study_uid}): {moving_reason}")
            patient_result["skip_reason"] = " ; ".join(reasons)
            warnings.append(f"{patient_id}: {patient_result['skip_reason']}")
            selection[patient_id] = patient_result
            continue

        patient_result["status"] = "ready"
        patient_result["fixed"] = {
            "study_uid": fixed_study_uid,
            "study_date": study_date_map[fixed_study_uid],
            "series_uid": fixed_choice.series_uid,
            "series_path": str(fixed_choice.series_path),
            "series_description": fixed_choice.series_description,
            "slice_count": fixed_choice.slice_count,
        }
        patient_result["moving"] = {
            "study_uid": moving_study_uid,
            "study_date": study_date_map[moving_study_uid],
            "series_uid": moving_choice.series_uid,
            "series_path": str(moving_choice.series_path),
            "series_description": moving_choice.series_description,
            "slice_count": moving_choice.slice_count,
        }
        patient_result["reasoning"] = f"FIXED: {fixed_reason} MOVING: {moving_reason}"
        selection[patient_id] = patient_result

    return selection


def create_output_structure(source_root: Path, dataset_root: Path, warnings: List[str]) -> None:
    (dataset_root / "raw").mkdir(parents=True, exist_ok=True)
    (dataset_root / "processed").mkdir(parents=True, exist_ok=True)
    (dataset_root / "transforms").mkdir(parents=True, exist_ok=True)
    (dataset_root / "results").mkdir(parents=True, exist_ok=True)

    if RAW_LINK.exists() or RAW_LINK.is_symlink():
        if RAW_LINK.is_symlink():
            current_target = RAW_LINK.resolve(strict=False)
            if current_target != source_root.resolve():
                warnings.append(
                    f"raw symlink exists but points to '{current_target}', expected '{source_root.resolve()}'."
                )
        else:
            warnings.append(
                f"Cannot create raw symlink at {RAW_LINK}: path exists and is not a symlink."
            )
    else:
        RAW_LINK.symlink_to(Path("../../qin_prostate_repeatability"))


def convert_dicom_series_to_nifti(series_dir: Path, output_nifti: Path) -> None:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not files:
        raise RuntimeError(f"No DICOM series files resolved by GDCM in {series_dir}")
    reader.SetFileNames(files)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_nifti))


def patient_numeric_suffix(patient_id: str) -> str:
    if "-" in patient_id:
        return patient_id.split("-")[-1]
    return patient_id


def process_patients(
    selection: Dict[str, Dict], dataset_root: Path, warnings: List[str]
) -> Tuple[List[Dict[str, str]], Dict[str, Dict]]:
    metadata_rows: List[Dict[str, str]] = []
    outputs: Dict[str, Dict] = {}

    processed_root = dataset_root / "processed"
    for patient_id, info in selection.items():
        outputs[patient_id] = {"status": info["status"], "fixed_path": "", "moving_path": ""}
        if info["status"] != "ready":
            continue

        suffix = patient_numeric_suffix(patient_id)
        patient_out_dir = processed_root / f"patient_{suffix}"
        patient_out_dir.mkdir(parents=True, exist_ok=True)

        fixed_nifti = patient_out_dir / "fixed.nii.gz"
        moving_nifti = patient_out_dir / "moving.nii.gz"

        fixed_series = Path(info["fixed"]["series_path"])
        moving_series = Path(info["moving"]["series_path"])

        try:
            convert_dicom_series_to_nifti(fixed_series, fixed_nifti)
            convert_dicom_series_to_nifti(moving_series, moving_nifti)
        except Exception as exc:  # pylint: disable=broad-except
            warnings.append(f"{patient_id}: conversion failed ({exc}).")
            info["status"] = "skipped"
            info["skip_reason"] = f"conversion failed: {exc}"
            outputs[patient_id]["status"] = "skipped"
            continue

        outputs[patient_id] = {
            "status": "processed",
            "fixed_path": str(fixed_nifti),
            "moving_path": str(moving_nifti),
        }

        metadata_rows.append(
            {
                "patient_id": patient_id,
                "fixed_path": str(fixed_nifti),
                "moving_path": str(moving_nifti),
                "fixed_series_description": info["fixed"]["series_description"],
                "moving_series_description": info["moving"]["series_description"],
                "study_date_fixed": info["fixed"]["study_date"],
                "study_date_moving": info["moving"]["study_date"],
            }
        )

    return metadata_rows, outputs


def write_metadata_csv(metadata_rows: List[Dict[str, str]], dataset_root: Path) -> Path:
    out_csv = dataset_root / "metadata.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow(row)
    return out_csv


def write_results(
    dataset_root: Path, discovery: Dict[str, Dict], selection: Dict[str, Dict], outputs: Dict[str, Dict], warnings: List[str]
) -> Tuple[Path, Path]:
    results_dir = dataset_root / "results"
    report_path = results_dir / "selection_report.json"
    warnings_path = results_dir / "warnings.log"

    report_patients = []
    for patient_id, patient_entry in discovery.items():
        studies_payload = []
        for study_uid, study_entry in patient_entry["studies"].items():
            counts = {
                "MR": len(study_entry["series_by_type"]["MR"]),
                "SEG": len(study_entry["series_by_type"]["SEG"]),
                "SR": len(study_entry["series_by_type"]["SR"]),
                "OTHER": len(study_entry["series_by_type"]["OTHER"]),
            }
            mr_meta_payload = []
            for mr in study_entry["mr_series_meta"]:
                mr_meta_payload.append(
                    {
                        "series_uid": mr.series_uid,
                        "series_path": str(mr.series_path),
                        "study_date": mr.study_date,
                        "series_description": mr.series_description,
                        "modality": mr.modality,
                        "sequence_name": mr.sequence_name,
                        "spacing_proxy": mr.spacing_proxy,
                        "slice_count": mr.slice_count,
                    }
                )

            studies_payload.append(
                {
                    "study_uid": study_uid,
                    "series_counts": counts,
                    "mr_series": mr_meta_payload,
                }
            )

        report_patients.append(
            {
                "patient_id": patient_id,
                "num_studies": len(patient_entry["studies"]),
                "study_dates": selection.get(patient_id, {}).get("study_dates", {}),
                "selection": selection.get(patient_id, {}),
                "outputs": outputs.get(patient_id, {}),
                "studies": studies_payload,
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "dataset_root": str(DATASET_ROOT),
        "include_keywords": INCLUDE_KEYWORDS,
        "exclude_keywords": EXCLUDE_KEYWORDS,
        "warnings_count": len(warnings),
        "warnings": warnings,
        "patients": report_patients,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with warnings_path.open("w", encoding="utf-8") as f:
        for warning in warnings:
            f.write(warning + "\n")

    return report_path, warnings_path


def print_report(
    discovery: Dict[str, Dict], selection: Dict[str, Dict], outputs: Dict[str, Dict], metadata_csv_path: Path
) -> None:
    print("=== QIN-Prostate Repeatability Preparation Report ===")
    print(f"Detected patients: {len(discovery)}")
    for patient_id in sorted(discovery.keys()):
        patient_entry = discovery[patient_id]
        sel = selection.get(patient_id, {})
        out = outputs.get(patient_id, {})
        print(f"\nPatient: {patient_id}")
        print(f"  Study count: {len(patient_entry['studies'])}")
        print("  Study dates:")
        if sel.get("study_dates"):
            for study_uid, study_date in sorted(sel["study_dates"].items(), key=lambda x: x[1]):
                print(f"    - {study_uid}: {study_date}")
        else:
            print("    - <none>")

        if sel.get("status") == "ready" or out.get("status") == "processed":
            print("  Selected T2 series:")
            print(
                f"    - FIXED  ({sel['fixed']['study_date']}): {sel['fixed']['series_uid']} "
                f"| {sel['fixed']['series_description']}"
            )
            print(
                f"    - MOVING ({sel['moving']['study_date']}): {sel['moving']['series_uid']} "
                f"| {sel['moving']['series_description']}"
            )
            print(f"  Reasoning: {sel.get('reasoning', '')}")
            print("  Output files:")
            print(f"    - {out.get('fixed_path', '')}")
            print(f"    - {out.get('moving_path', '')}")
        else:
            print(f"  Status: skipped")
            print(f"  Reason: {sel.get('skip_reason', '<unknown>')}")

    print(f"\nmetadata.csv: {metadata_csv_path}")


def main() -> None:
    warnings: List[str] = []

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {SOURCE_ROOT}")

    create_output_structure(SOURCE_ROOT, DATASET_ROOT, warnings)
    discovery = discover_dataset(SOURCE_ROOT, warnings)
    _ = extract_mr_metadata(discovery, warnings)
    selection = select_registration_pairs(discovery, warnings)
    metadata_rows, outputs = process_patients(selection, DATASET_ROOT, warnings)
    metadata_csv_path = write_metadata_csv(metadata_rows, DATASET_ROOT)
    report_path, warnings_path = write_results(DATASET_ROOT, discovery, selection, outputs, warnings)
    print_report(discovery, selection, outputs, metadata_csv_path)
    print(f"selection_report.json: {report_path}")
    print(f"warnings.log: {warnings_path}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
