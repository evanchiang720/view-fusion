import tarfile
import os

# ShapeNet Standard 13 Categories mapping
SYNSET_DICT = {
    "02691156": "airplane",
    "02828884": "bench",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03211117": "display",
    "03636649": "lamp",
    "03691459": "speaker",
    "04090263": "rifle",
    "04256520": "sofa",
    "04379243": "table",
    "04401088": "telephone",
    "04530566": "vessel/boat"
}

shard_path = "data/nmr/NMR_sharded_100_4/NMR-test-04.tar"
if not os.path.exists(shard_path):
    shard_path = "/home/chc302/public/users/chc302/NMR_sharded_100_4/NMR-test-01.tar"

print(f"[*] Scanning shard: {shard_path} ...\n")

current_category = None
object_idx = -1
img_count_in_obj = 0

print(f"{'Object Index':<15} | {'Category Name':<15} | {'Category ID':<12} | {'Sample Path'}")
print("-" * 80)

with tarfile.open(shard_path, "r") as tar:
    for member in tar.getmembers():
        if member.name.endswith(".png") or member.name.endswith(".jpg"):
            if img_count_in_obj % 24 == 0:
                object_idx += 1
                # Detect category ID from file path
                found_cat = "unknown"
                cat_id_found = "unknown"
                for cat_id, cat_name in SYNSET_DICT.items():
                    if cat_id in member.name:
                        found_cat = cat_name
                        cat_id_found = cat_id
                        break
                
                # Print when a new category appears
                if found_cat != current_category:
                    current_category = found_cat
                    print(f"Index [{object_idx:<6}]  | {found_cat:<15} | {cat_id_found:<12} | {member.name.split('/')[-2] if '/' in member.name else member.name}")

            img_count_in_obj += 1

print("-" * 80)
print(f"[*] Total 3D objects in this shard: {object_idx + 1}")
