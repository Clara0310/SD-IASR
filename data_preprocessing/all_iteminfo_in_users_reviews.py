import json

def extract_ordered_metadata(user_review_path, metadata_path, output_name):
    # Step 1：讀取使用者評論（已排序的 JSON）
    with open(user_review_path, "r", encoding="utf-8") as f:
        user_reviews = json.load(f)

    # 按照順序取出 asin（商品可能重複！）
    ordered_asins = [review["asin"] for review in user_reviews]
    print(f"使用者評論共 {len(ordered_asins)} 次（含重複 asin）")

    # Step 2：把 metadata 全部讀進 dictionary（快速查詢）
    asin_to_meta = {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            asin = obj.get("asin")
            if asin:
                asin_to_meta[asin] = obj

    print(f"Metadata 中共 {len(asin_to_meta)} 個商品可查詢")

    # Step 3：依 user 的順序建立商品 metadata（可重複）
    ordered_metadata = []

    for asin in ordered_asins:
        if asin in asin_to_meta:
            ordered_metadata.append(asin_to_meta[asin])
        else:
            # 如果 metadata 缺少該商品，把空資訊補上（避免錯誤）
            ordered_metadata.append({"asin": asin, "metadata_missing": True})

    print(f"最終輸出商品資訊筆數：{len(ordered_metadata)}")

    # Step 4：輸出成 JSON（list 格式，順序固定）
    output_path = f"{output_name}_items_ordered.json"
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(ordered_metadata, out, indent=4, ensure_ascii=False)

    print(f"📁 已輸出排序後商品資訊到：{output_path}")


if __name__ == "__main__":
    user_review_path = "/home/clara_r76121188/thesis/A1006HCQDMYC5W_reviews_sorted.json"
    metadata_path = "/home/clara_r76121188/thesis/SD-IASR/datasets/meta_Grocery_and_Gourmet_Food.json"

    extract_ordered_metadata(
        user_review_path,
        metadata_path,
        output_name="A1006HCQDMYC5W"
    )
