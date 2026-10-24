# SaltCare — Streamlit Community Cloud

## File utama
`streamlit_app.py`

## Jalankan lokal
```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Deploy ke Streamlit Community Cloud
1. Commit dan push folder proyek ke GitHub.
2. Buka https://share.streamlit.io/ dan pilih repository SaltCare.
3. Pilih branch `main` dan main file path `streamlit_app.py`.
4. Deploy dan periksa log bila ada dependensi yang gagal dipasang.

## Catatan penting
- Model `models/random_forest_best.joblib` dipertahankan; aplikasi tidak melatih model baru.
- Fitur input, mapping kategori, urutan fitur, label target, dan metadata mengikuti artefak serta kode Flask yang diberikan.
- Riwayat memakai SQLite di `instance/saltcare_streamlit.db`. Pada Streamlit Community Cloud, filesystem lokal tidak boleh dianggap permanen; riwayat dapat hilang saat aplikasi diredeploy atau instance diganti/di-reset. Untuk riwayat permanen, sambungkan database eksternal persisten pada tahap berikutnya.
- SHAP memeriksa additivity terhadap probabilitas kelas. Jika tidak cocok, aplikasi menandai skala output sebagai belum terverifikasi.
- Jangan gunakan hasil aplikasi sebagai diagnosis medis.
