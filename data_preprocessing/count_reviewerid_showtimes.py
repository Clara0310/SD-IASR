import json
from collections import Counter
import csv


# -------------------------------------------------------
# 功能 1：計算每個 user 出現的次數，並統計有多少 user 出現相同次數
# -------------------------------------------------------
def user_frequency(path, dataset_name):
    counter = Counter()

    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            user = obj.get("reviewerID")
            if user:
                counter[user] += 1

    freq_count = Counter(counter.values())

    output_path = f"{dataset_name}_user_frequency.csv"

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["review_times", "user_count"])

        for times in sorted(freq_count):
            print(f"review 次數 = {times}, user 數量 = {freq_count[times]}")
            writer.writerow([times, freq_count[times]])

    print(f"\n📁 CSV 已輸出到: {output_path}\n")
    

# -------------------------------------------------------
# 功能 2：找出「評論剛好 10 次」的 reviewer，從中挑出 10 個
# -------------------------------------------------------
def sample_users_with_10_reviews(path, dataset_name):
    counter = Counter()

    # 先統計每個 user 的評論次數
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            user = obj.get("reviewerID")
            if user:
                counter[user] += 1

    # 篩選出「剛好評論 10 次」的 reviewer
    users_10 = [user for user, count in counter.items() if count == 10]

    # 為了結果穩定，照 reviewerID 排序後取前 10 個
    users_10_sorted = sorted(users_10)
    picked_users = users_10_sorted[:10]

    print(f"\n--- {dataset_name}：評論剛好 10 次的 reviewer（其中 10 位）---")
    for i, user in enumerate(picked_users, start=1):
        print(f"{i}. reviewerID = {user}，reviews = 10")

    # 匯出成 CSV
    output_path = f"{dataset_name}_10reviews_sample10.csv"
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "reviewerID", "review_count"])
        for i, user in enumerate(picked_users, start=1):
            writer.writerow([i, user, 10])

    print(f"📁 已輸出 10 位 reviewer（剛好 10 則評論）到: {output_path}\n")
    # 計算每個 user 的評論次數
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            user = obj.get("reviewerID")
            if user:
                counter[user] += 1
                

# -------------------------------------------------------
# 功能3：找出「評論剛好 20 次」的 reviewer，從中挑出 2 位
# -------------------------------------------------------
def sample_users_with_20_reviews(path, dataset_name):
    counter = Counter()

    # 先統計每個 user 的評論次數
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            user = obj.get("reviewerID")
            if user:
                counter[user] += 1

    # 篩選出「剛好評論 20 次」的 reviewer
    users_20 = [user for user, count in counter.items() if count == 20]

    # 為了結果穩定，照 reviewerID 排序後取前 2 位
    users_20_sorted = sorted(users_20)
    picked_users = users_20_sorted[:2]

    print(f"\n--- {dataset_name}：評論剛好 20 次的 reviewer（其中 2 位）---")
    for i, user in enumerate(picked_users, start=1):
        print(f"{i}. reviewerID = {user}，reviews = 20")

    # 匯出成 CSV
    output_path = f"{dataset_name}_20reviews_sample2.csv"
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "reviewerID", "review_count"])
        for i, user in enumerate(picked_users, start=1):
            writer.writerow([i, user, 20])

    print(f"📁 已輸出 2 位 reviewer（剛好 20 則評論）到: {output_path}\n")


    

# -------------------------------------------------------
# 主程式
# -------------------------------------------------------
if __name__ == '__main__':
    
    #計算每個 user 出現的次數，並統計有多少 user 出現相同次數，最後輸出成 csv 檔案
    path_appliances = "/home/clara_r76121188/thesis/SD-IASR/datasets/Appliances.json"
    path_GroceryandFood = "/home/clara_r76121188/thesis/SD-IASR/datasets/Grocery_and_Gourmet_Food.json"
    path_homeandkitchen = "/home/clara_r76121188/thesis/SD-IASR/datasets/Home_and_Kitchen.json"
    
    # 各 dataset 的 histogram
    # user_frequency(path_appliances,"appliances")
    # user_frequency(path_GroceryandFood,"GroceryandFood")
    # user_frequency(path_homeandkitchen,"homeandkitchen")

    #從「評論次數剛好 10 次」的 reviewer 裡各挑 10 位
    # sample_users_with_10_reviews(path_appliances, "appliances")
    # sample_users_with_10_reviews(path_GroceryandFood, "GroceryandFood")
    # sample_users_with_10_reviews(path_homeandkitchen, "homeandkitchen")
    
    #從「評論次數剛好 20 次」的 reviewer 裡各挑 2 位
    sample_users_with_20_reviews(path_appliances, "appliances")
    sample_users_with_20_reviews(path_GroceryandFood, "GroceryandFood")
    sample_users_with_20_reviews(path_homeandkitchen, "homeandkitchen")