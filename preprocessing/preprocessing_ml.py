import pandas as pd
import os


# ============================================================
# 1. LOKASI FILE
# ============================================================

FILE_INPUT = "../webscraper/mbg_dataset_ml.csv"
FILE_OUTPUT = "mbg_dataset_preprocessed.csv"


# ============================================================
# 2. MEMBACA DATASET
# ============================================================

if not os.path.exists(FILE_INPUT):
    print(f"ERROR: File tidak ditemukan -> {FILE_INPUT}")
    exit()

df = pd.read_csv(
    FILE_INPUT,
    sep=";"
)


print("=" * 70)
print("PREPROCESSING DATASET MACHINE LEARNING")
print("=" * 70)


# ============================================================
# 3. INFORMASI DATASET AWAL
# ============================================================

print("\nJumlah data awal:")
print(len(df))

print("\nKolom dataset:")
print(df.columns.tolist())


# ============================================================
# 4. MEMASTIKAN TIPE DATA
# ============================================================

df["tanggal_kasus"] = pd.to_datetime(
    df["tanggal_kasus"],
    errors="coerce"
)

df["jumlah_kasus"] = pd.to_numeric(
    df["jumlah_kasus"],
    errors="coerce"
)


# ============================================================
# 5. MEMBUAT FITUR WAKTU
# ============================================================

# Tanggal tidak digunakan langsung sebagai fitur.
# Tanggal diubah menjadi tahun dan bulan.

df["tahun"] = df["tanggal_kasus"].dt.year
df["bulan"] = df["tanggal_kasus"].dt.month


# ============================================================
# 6. MENENTUKAN TARGET (Y)
# ============================================================

# Y adalah nilai yang ingin diprediksi.

y = df["jumlah_kasus"]


# ============================================================
# 7. MENENTUKAN FITUR (X)
# ============================================================

# Fitur yang digunakan:
# - tahun
# - bulan
# - provinsi

X = df[
    [
        "tahun",
        "bulan",
        "provinsi"
    ]
].copy()


# ============================================================
# 8. ENCODING PROVINSI
# ============================================================

# Provinsi masih berupa teks.
# Machine Learning membutuhkan data numerik.
#
# One-Hot Encoding digunakan untuk mengubah
# setiap kategori provinsi menjadi kolom 0 dan 1.

X = pd.get_dummies(
    X,
    columns=["provinsi"],
    dtype=int
)


# ============================================================
# 9. MEMASTIKAN DATA X DAN Y LENGKAP
# ============================================================

data_model = X.copy()

data_model["jumlah_kasus"] = y.values

data_model = data_model.dropna()


# Pisahkan kembali X dan Y setelah pengecekan missing value.

X = data_model.drop(
    columns=["jumlah_kasus"]
)

y = data_model["jumlah_kasus"]


# ============================================================
# 10. MENAMPILKAN TARGET (Y)
# ============================================================

print("\n" + "=" * 70)
print("TARGET (Y)")
print("=" * 70)

print(y.to_string(index=False))


# ============================================================
# 11. MENAMPILKAN FITUR (X)
# ============================================================

print("\n" + "=" * 70)
print("FITUR (X)")
print("=" * 70)

print(X.to_string(index=False))


# ============================================================
# 12. MENGGABUNGKAN X DAN Y
# ============================================================

df_preprocessed = X.copy()

df_preprocessed["jumlah_kasus"] = y.values


# ============================================================
# 13. MENYIMPAN DATASET HASIL PREPROCESSING
# ============================================================

df_preprocessed.to_csv(
    FILE_OUTPUT,
    index=False,
    encoding="utf-8-sig",
    sep=";"
)


# ============================================================
# 14. INFORMASI HASIL PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("HASIL PREPROCESSING")
print("=" * 70)

print(
    f"Jumlah data       : {len(df_preprocessed)}"
)

print(
    f"Jumlah fitur (X)  : {len(X.columns)}"
)

print(
    "Target (Y)        : jumlah_kasus"
)


# ============================================================
# 15. DAFTAR FITUR
# ============================================================

print("\nKolom fitur (X):")

for kolom in X.columns:
    print(f"- {kolom}")


print("\nKolom target (Y):")
print("- jumlah_kasus")


# ============================================================
# 16. FILE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILE OUTPUT")
print("=" * 70)

print(FILE_OUTPUT)

print("\nPreprocessing selesai.")