// ...existing code...
# LSTM Dataset Project (Dataset_LSTM)

Ringkasan singkat dokumentasi untuk skrip pelatihan LSTM yang ada di workspace.

**Author: I DEWA GEDE MAHESTA PARAWANGSA**

## Struktur proyek
- [TrainLSTM1.py](TrainLSTM1.py) — Skenario 1 (single LSTM).
  - Variabel konfigurasi utama: [`dataset_dir`](TrainLSTM1.py), fungsi label: [`enc_lbl`](TrainLSTM1.py)
- [TrainLSTM2.py](TrainLSTM2.py) — Skenario 2 (LSTM stacked 64→32).
  - Variabel konfigurasi utama: [`dataset_dir`](TrainLSTM2.py), fungsi label: [`enc_lbl`](TrainLSTM2.py)
- [TrainLSTM3.py](TrainLSTM3.py) — Skenario 3 (LSTM stacked 32→32).
  - Variabel konfigurasi utama: [`dataset_dir`](TrainLSTM3.py), fungsi label: [`enc_lbl`](TrainLSTM3.py)
- Data folders:
  - [CI_Test/](CI_Test/)
  - [CI_Test_6Class/](CI_Test_6Class/)
  - [CS_Train/](CS_Train/)
  - [Arsip/](Arsip/)
- Konfigurasi / metadata:
  - [.gitignore](.gitignore)
  - [requirements.txt](requirements.txt)

## Persyaratan
Instal dependensi yang tercantum di [requirements.txt](requirements.txt):
```sh
pip install -r requirements.txt
```

## Penyiapan data
- Pastikan file .mat peserta berada di folder yang sesuai (default masing‑masing skrip menunjuk ke `CS_Train` via [`dataset_dir`](TrainLSTM1.py)).
- Jika ingin mengabaikan file .mat di repo, periksa [`.gitignore`](.gitignore) dan jika file .mat sudah ter-track: gunakan `git rm --cached <path>` lalu commit. Pertimbangkan Git LFS untuk file besar.

## Menjalankan pelatihan
Jalankan salah satu skrip dari terminal:
```sh
python TrainLSTM1.py   # Skenario 1
python TrainLSTM2.py   # Skenario 2
python TrainLSTM3.py   # Skenario 3
```
- Output: grafik (matplotlib), confusion matrix, dan ringkasan metrik per-file & ringkasan antar-partisipan di stdout.
- Periksa parameter seperti batch_size, epochs, dan early stopping di tiap skrip.

## Penjelasan singkat skrip
- Semua skrip:
  - Membaca .mat dengan key `data`, `valence_labels`, `arousal_labels`, `dominance_labels`.
  - Mengubah bentuk X ke (12, 5, 64) dan menggabungkan label 3-bit menjadi kelas aksara via [`enc_lbl`](TrainLSTM1.py).
  - Melakukan 10‑fold KFold CV dan mencetak metrik: accuracy, precision, recall, F1, epoch stats.
- Perbedaan skenario:
  - [TrainLSTM1.py](TrainLSTM1.py): single LSTM layer.
  - [TrainLSTM2.py](TrainLSTM2.py): stacked LSTM 64 → 32.
  - [TrainLSTM3.py](TrainLSTM3.py): stacked LSTM 32 → 32 + dropout.