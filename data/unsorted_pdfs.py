import os
import shutil

# Source folder with unsorted PDFs
source_folder = r"D:\CODING\Machine Learning\Architect-RAG-LLM-Assistant\unsorted_pdfs"

# Destination base folder
target_base = r"D:\CODING\Machine Learning\Architect-RAG-LLM-Assistant\data"

# Exact mapping from filename (without .pdf) → folder
file_mapping = {
    # building_codes
    "UFGS COMPLETE": "building_codes",
    "macdill_design_guide_2025_c1": "building_codes",
    "p100_2010": "building_codes",
    "P100_2021_Addendum_2022": "building_codes",
    "ufc_3_101_01_2020_c4": "building_codes",
    "ufc_3_301_01_2023_c4": "building_codes",
    "ufc_3_130_04_2025": "building_codes",
    "in.gov.nbc.2016.vol1.digital": "building_codes",
    "The_Merged_Approved_Documents_Oct24": "building_codes",

    # case_studies
    "Dynamic Urban Planning- an Agent-Based Model Coupling Mobility Mode and": "case_studies",
    "predictSLUMS- A new model for identifying and predicting informal": "case_studies",
    "Reimagining City Configuration- Automated Urban Planning via Adversarial": "case_studies",
    "Mapping Inequalities in Activity-based Carbon Footprints of Urban": "case_studies",
    "Queue Routing Strategies to Improve Equitable Housing Coordination in": "case_studies",
    "Spatial Structure of City Population Growth": "case_studies",
    "The benefits and costs of agglomeration- insights from economics and": "case_studies",
    "Towards AI Urban Planner in the Age of GenAI, LLMs, and Agentic AI": "case_studies",

    # material_guide
    "Innovative-Construction-Technologies": "material_guide",
    "innovative-housing-design": "material_guide",
    "Keys_to_Innovative_Housing_Design_in_Architecture": "material_guide",
    "Construction Technology - An Illustrated Introduction": "material_guide",
    "BUILDING-MATERIALS-AND-CONSTRUCTION": "material_guide",
    "dg_small_house_model": "material_guide",
    "bedg_agwsg": "material_guide",

    # misc
    "Understanding_Sustainable_Architecture": "misc",
    "L-G-0003911789-0008043387": "misc",
    "25845916": "misc",
    "G92360811922": "misc",
    "isarc2004-IL03": "misc",
    "Architecture and Planning (AR)": "misc",
    "Architecture-and-Planning": "misc",
    "UrbanPlanning_new": "misc",
    "e9b3296379c32130849b6d25133cb7d5_MIT4_607F09_lec06": "misc",
    "IJRPR31256": "misc",
    "UJCRT2012187": "misc"
}

# Create destination folders if they don't exist
for folder in set(file_mapping.values()):
    os.makedirs(os.path.join(target_base, folder), exist_ok=True)

# Move files
for file in os.listdir(source_folder):
    if not file.lower().endswith(".pdf"):
        continue

    name_without_ext = os.path.splitext(file)[0]
    if name_without_ext in file_mapping:
        dest_folder = file_mapping[name_without_ext]
        shutil.move(
            os.path.join(source_folder, file),
            os.path.join(target_base, dest_folder, file)
        )
        print(f"Moved: {file} → {dest_folder}")
    else:
        print(f"⚠ No mapping for: {file}")

print("✅ Sorting complete.")
