import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
import os
import sys

def preprocess_and_save(input_path, output_path):
    df = pd.read_csv(input_path)

    # Replace value & drop kolom unik
    df['status_nikah'] = df['status_nikah'].replace('Divorced', 'Single')
    df.replace('Unknown', np.nan, inplace=True)
    df.drop(columns=['client_id'], inplace=True)

    # Age Binning
    age_binning = [19, 59, 100]
    age_category = ['dewasa', 'lansia']
    df['usia'] = pd.cut(df['usia'], bins=age_binning, labels=age_category)
    df['usia'] = df['usia'].astype(str)

    # Imputasi nilai kategorikal dengan modus
    imputer = SimpleImputer(strategy='most_frequent')
    for col in ['pendidikan', 'status_nikah', 'penghasilan_tahunan']:
        df[col] = imputer.fit_transform(df[[col]]).ravel()

    # Normalisasi
    scaler = MinMaxScaler()
    num_columns = ['jumlah_tanggungan', 'lama_nasabah', 'jumlah_produk', 'bulan_nonactive',
                   'jumlah_kontak', 'total_limit_kredit', 'total_limit_kredit_dipakai',
                   'sisa_limit_kredit', 'rasio_transaksi_Q4_Q1 ', 'total_transaksi',
                   'jumlah_transaksi', 'rasio_jumlah_transaksi_Q4_Q1', 'rasio_pemakaian']
    df[num_columns] = scaler.fit_transform(df[num_columns])

    # OneHot Encoding fitur kategorikal nominal
    cat_columns = ['gender', 'status_nikah']
    ohe = OneHotEncoder(sparse_output=False)
    encoded_array = ohe.fit_transform(df[cat_columns])
    encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(cat_columns), index=df.index)
    df.drop(columns=cat_columns, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)

    # Ordinal Encoding fitur kategorikal ordinal
    cat_ordinal_columns = ['usia', 'pendidikan', 'penghasilan_tahunan', 'tipe_kartu_kredit']
    OE = OrdinalEncoder(
        categories=[
            ['dewasa', 'lansia'],
            ['Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'],
            ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'],
            ['Blue', 'Silver', 'Gold', 'Platinum']
        ],
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    df[cat_ordinal_columns] = OE.fit_transform(df[cat_ordinal_columns])

    # Pastikan folder tujuan ada, jika tidak buat dulu
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder '{output_dir}' dibuat karena belum ada.")

    # Simpan file hasil preprocessing
    df.to_csv(output_path, index=False)
    print(f"File preprocessing disimpan di: {output_path}")

if __name__ == "__main__":
    print("Script preprocessing mulai...")
    print("Argumen:", sys.argv)

    if len(sys.argv) != 3:
        print("Gunakan: python automate_Nama-siswa.py path_input path_output")
    else:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        print(f"Load data dari: {input_csv}")
        print(f"Simpan hasil ke: {output_csv}")
        preprocess_and_save(input_csv, output_csv)
        print("Proses preprocessing selesai.")