import json
from datetime import datetime


def find_reviews_by_user(path, target_user):
    results = []

    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            user = obj.get("reviewerID")

            if user == target_user:
                results.append(obj)

    # 如果沒有評論
    if not results:
        print(f"⚠ 找不到 reviewerID = {target_user} 的評論")
        return

    # ⭐ 按照 unixReviewTime 排序（由舊到新）
    results.sort(key=lambda x: x.get("unixReviewTime", 0))

    print(f"找到 {len(results)} 筆評論屬於 reviewerID = {target_user}")

    # 印出排序後的每筆評論
    for i, item in enumerate(results, start=1):
        readable_time = datetime.fromtimestamp(item["unixReviewTime"]).strftime("%Y-%m-%d")
        print(f"\n--- Review #{i} ---  日期：{readable_time}")
        print(json.dumps(item, indent=4, ensure_ascii=False))

    # ⭐輸出成標準 JSON（是 list，已排序）
    output_path = f"{target_user}_reviews_sorted.json"
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(results, out, indent=4, ensure_ascii=False)

    print(f"\n📁 已輸出排序後的 JSON 檔案到：{output_path}")


if __name__ == "__main__":
    #path = "/home/clara_r76121188/thesis/SD-IASR/datasets/Appliances.json"
    path = "/home/clara_r76121188/thesis/SD-IASR/datasets/Grocery_and_Gourmet_Food.json"
    #path = "/home/clara_r76121188/thesis/SD-IASR/datasets/Home_and_Kitchen.json"
    
    target_user = "A1006HCQDMYC5W"  # 替換成你想查找的 reviewerID

    find_reviews_by_user(path, target_user)
