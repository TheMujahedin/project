import glob
import re
import matplotlib.pyplot as plt
import pandas as pd

# 1. Ανάγνωση όλων των αρχείων αποτελεσμάτων
files = glob.glob("res_*.txt")
records = []

for filepath in files:
    match = re.search(r"res_([a-zA-Z]+)_D(\d+)\.txt", filepath)
    if not match:
        continue
    
    method, d_val = match.groups()
    d_val = int(d_val)
    
    r5, r10, map10, qtime = None, None, None, None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Mean Recall@5:"):
                r5 = float(line.split(":")[1].strip())
            elif line.startswith("Mean Recall@10:"):
                r10 = float(line.split(":")[1].strip())
            elif line.startswith("mAP@10:"):
                map10 = float(line.split(":")[1].strip())
            elif line.startswith("Average query time:"):
                qtime = float(line.split(":")[1].strip())
                
    if r5 is not None:
        records.append({
            "Method": method.upper(),
            "D": d_val,
            "Recall@5": r5,
            "Recall@10": r10,
            "mAP@10": map10,
            "QueryTime_ms": qtime
        })

df = pd.DataFrame(records)
if df.empty:
    print("Δεν βρέθηκαν ακόμα αρχεία αποτελεσμάτων res_*.txt.")
    exit()

# Ταξινόμηση και αποθήκευση σε CSV
df = df.sort_values(by=["Method", "D"])
df.to_csv("results_summary.csv", index=False)

print("\n=== ΣΥΓΚΕΝΤΡΩΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ===")
print(df.to_string(index=False))

# 2. Δημιουργία Γραφήματος mAP@10 vs D
plt.figure(figsize=(9, 5))
for method, group in df.groupby("Method"):
    plt.plot(group["D"], group["mAP@10"], marker='o', label=method)
plt.title("mAP@10 ανάλογα με τον αριθμό Distractors (D)")
plt.xlabel("Αριθμός Distractors (D)")
plt.ylabel("mAP@10")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("plot_map10_vs_D.png", dpi=300)
plt.close()

# 3. Δημιουργία Γραφήματος Average Query Time vs D
plt.figure(figsize=(9, 5))
for method, group in df.groupby("Method"):
    plt.plot(group["D"], group["QueryTime_ms"], marker='s', label=method)
plt.title("Μέσος Χρόνος Αναζήτησης (ms) ανάλογα με το D")
plt.xlabel("Αριθμός Distractors (D)")
plt.ylabel("Query Time (ms)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("plot_time_vs_D.png", dpi=300)
plt.close()

print("\nΤα διαγράμματα 'plot_map10_vs_D.png' και 'plot_time_vs_D.png' δημιουργήθηκαν επιτυχώς!")