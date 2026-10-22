"""
Data Mining - Eksplorasi Dataset Travel & Tourism
Dataset : Travel And Tourism.csv
Library : pandas, numpy, matplotlib

Isi script:
  [0] Data Loading
  [1] Eksplorasi Data: Informasi umum
  [2] Poin 1 - Pembagian jenis data (Kategorikal & Numerikal)
  [3] Poin 2 - Tipe data kolom kategorikal
  [4] Poin 3 - Jumlah data duplicate & null
  [5] Poin 4 - Penanganan outlier
  [6] Poin 4 - Encoding data
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILE_PATH = "Travel And Tourism.csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


# ======================================================================
# [0] DATA LOADING
# ======================================================================
judul("[0] DATA LOADING")

df = pd.read_csv(FILE_PATH)

print("Versi pandas :", pd.__version__)
print("Jumlah baris :", df.shape[0])
print("Jumlah kolom :", df.shape[1])
print("\n5 data teratas:")
print(df.head())


# ======================================================================
# [1] EKSPLORASI DATA - INFORMASI UMUM
# ======================================================================
judul("[1] EKSPLORASI DATA - INFORMASI UMUM")

print("Info dataset:")
df.info()

print("\nStatistik deskriptif (numerikal):")
print(df.describe().T)


# ======================================================================
# [2] POIN 1 - BAGI JENIS DATA (KATEGORIKAL & NUMERIKAL)
# ======================================================================
judul("[2] POIN 1 - JENIS DATA: KATEGORIKAL & NUMERIKAL")

kolom_numerikal = df.select_dtypes(include="number").columns.tolist()
kolom_kategorikal = df.select_dtypes(exclude="number").columns.tolist()

print(f"Data Numerikal ({len(kolom_numerikal)} kolom):")
for k in kolom_numerikal:
    print("  -", k)

print(f"\nData Kategorikal ({len(kolom_kategorikal)} kolom):")
for k in kolom_kategorikal:
    print("  -", k)

kolom_id_tanggal = [
    "Booking_ID", "Customer_ID", "Customer_Name",
    "Booking_Date", "Travel_Date", "Return_Date",
]
kolom_kategorikal_murni = [k for k in kolom_kategorikal if k not in kolom_id_tanggal]
print("\nKategorikal murni (di luar ID, nama, dan tanggal):")
print(kolom_kategorikal_murni)


# ======================================================================
# [3] POIN 2 - TIPE DATA KATEGORIKAL (float / int / object-str / dll)
# ======================================================================
judul("[3] POIN 2 - TIPE DATA KOLOM KATEGORIKAL")

tipe_kategorikal = df[kolom_kategorikal].dtypes
print(tipe_kategorikal)

print("\nRingkasan tipe data seluruh dataset:")
print(df.dtypes.value_counts())

print("\nJumlah nilai unik & contoh nilai kolom kategorikal murni:")
for k in kolom_kategorikal_murni:
    nilai = df[k].unique()
    print(f"  {k:<25} unik={len(nilai):<3} contoh={list(nilai[:4])}")

print("\nKesimpulan: kolom kategorikal bertipe teks (str/object), bukan float.")
print("Float hanya dipakai kolom numerikal seperti Hotel_Rating dan Customer_Rating.")


# ======================================================================
# [4] POIN 3 - JUMLAH DATA DUPLICATE & NULL
# ======================================================================
judul("[4] POIN 3 - DUPLICATE & NULL")

jumlah_duplikat = df.duplicated().sum()
print("Jumlah data duplicate (seluruh kolom):", jumlah_duplikat)

# Cek duplicate tanpa kolom ID unik (kemungkinan duplikat tersembunyi)
kolom_tanpa_id = df.columns.drop(["Booking_ID", "Customer_ID"])
print("Jumlah duplicate tanpa Booking_ID & Customer_ID:", df.duplicated(subset=kolom_tanpa_id).sum())

print("\nJumlah null per kolom:")
null_per_kolom = df.isnull().sum()
print(null_per_kolom)
print("\nTotal null di seluruh dataset:", int(null_per_kolom.sum()))

# Nilai 'Not Applicable' / 'Not Cancelled' bukan null, tetapi label yang sah.
print("\nCatatan: 'Not Applicable' pada Cancellation_Reason dihitung sebagai data valid,")
print("bukan null, karena artinya booking tidak dibatalkan.")
print(df["Cancellation_Reason"].value_counts())


# ======================================================================
# [5] POIN 4 - PENANGANAN OUTLIER (metode IQR)
# ======================================================================
judul("[5] POIN 4 - PENANGANAN OUTLIER (IQR)")

# Kolom ID tidak diperiksa karena bukan nilai yang bermakna secara statistik.
kolom_cek_outlier = [k for k in kolom_numerikal if k != "Booking_ID"]

batas = {}
ringkasan_outlier = []
for k in kolom_cek_outlier:
    q1 = df[k].quantile(0.25)
    q3 = df[k].quantile(0.75)
    iqr = q3 - q1
    bawah = q1 - 1.5 * iqr
    atas = q3 + 1.5 * iqr
    batas[k] = (bawah, atas)
    jumlah = int(((df[k] < bawah) | (df[k] > atas)).sum())
    ringkasan_outlier.append(
        {"Kolom": k, "Q1": q1, "Q3": q3, "Batas_Bawah": bawah,
         "Batas_Atas": atas, "Jumlah_Outlier": jumlah}
    )

tabel_outlier = pd.DataFrame(ringkasan_outlier)
print(tabel_outlier.to_string(index=False))

# Visualisasi boxplot
n = len(kolom_cek_outlier)
kolom_grid = 4
baris_grid = int(np.ceil(n / kolom_grid))
fig, axes = plt.subplots(baris_grid, kolom_grid, figsize=(16, 3.2 * baris_grid))
axes = axes.flatten()
for i, k in enumerate(kolom_cek_outlier):
    axes[i].boxplot(df[k])
    axes[i].set_title(k, fontsize=9)
for j in range(n, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.savefig("boxplot_outlier.png", dpi=120)
plt.close()
print("\nBoxplot disimpan ke: boxplot_outlier.png")

# Penanganan: capping (winsorizing) pada kolom yang punya outlier.
# Data asli tetap disimpan di df, hasil penanganan disimpan di df_clean.
df_clean = df.copy()
kolom_dengan_outlier = tabel_outlier.loc[tabel_outlier["Jumlah_Outlier"] > 0, "Kolom"].tolist()
print("\nKolom yang punya outlier:", kolom_dengan_outlier)

for k in kolom_dengan_outlier:
    bawah, atas = batas[k]
    df_clean[k] = df_clean[k].clip(lower=bawah, upper=atas)
    if k.startswith("Number_of"):
        df_clean[k] = df_clean[k].round().astype(int)

print("\nPerbandingan nilai maksimum sebelum dan sesudah capping:")
for k in kolom_dengan_outlier:
    print(f"  {k:<20} sebelum={df[k].max():<12} sesudah={df_clean[k].max()}")

print("\nCatatan: outlier tidak dihapus karena biaya perjalanan yang tinggi adalah")
print("hal wajar (perjalanan mewah/rombongan besar). Capping menjaga jumlah data tetap 1000.")


# ======================================================================
# [6] POIN 4 - ENCODING DATA
# ======================================================================
judul("[6] POIN 4 - ENCODING DATA")

# --- 6a. Label / ordinal encoding (kategori yang punya urutan atau hanya 2 nilai)
df_enc = df_clean.copy()

df_enc["Gender_Encoded"] = df_enc["Gender"].map({"Male": 0, "Female": 1})
df_enc["Cancellation_Status_Encoded"] = df_enc["Cancellation_Status"].map(
    {"Not Cancelled": 0, "Cancelled": 1}
)
# Meal plan punya urutan tingkat layanan, jadi cocok ordinal encoding
df_enc["Meal_Plan_Encoded"] = df_enc["Meal_Plan"].map(
    {"Room Only": 0, "Breakfast Included": 1, "Half Board": 2, "Full Board": 3}
)

print("Label/ordinal encoding:")
print(df_enc[["Gender", "Gender_Encoded",
              "Cancellation_Status", "Cancellation_Status_Encoded",
              "Meal_Plan", "Meal_Plan_Encoded"]].head())

# --- 6b. One-hot encoding (kategori nominal tanpa urutan, nilai unik sedikit)
kolom_onehot = ["Payment_Method", "Booking_Status", "Transportation_Type"]
df_enc = pd.get_dummies(df_enc, columns=kolom_onehot, dtype=int)

print("\nOne-hot encoding pada:", kolom_onehot)
kolom_baru = [c for c in df_enc.columns if c.startswith(tuple(kolom_onehot))]
print(df_enc[kolom_baru].head())

# --- 6c. Kolom yang tidak di-encode
kolom_tidak_encode = [
    "Booking_ID", "Customer_ID", "Customer_Name", "Booking_Date", "Travel_Date",
    "Return_Date", "Customer_City", "Customer_Country", "Destination_City",
    "Destination_Country", "Hotel_Name", "Cancellation_Reason", "Customer_Review_Comment",
]
print("\nKolom yang tidak di-encode (ID, tanggal, atau nilai unik terlalu banyak/teks bebas):")
print(kolom_tidak_encode)

print("\nUkuran data setelah encoding:", df_enc.shape)

# Simpan hasil akhir
df_enc.to_csv("Travel_And_Tourism_preprocessed.csv", index=False)
print("Hasil disimpan ke: Travel_And_Tourism_preprocessed.csv")

judul("SELESAI")